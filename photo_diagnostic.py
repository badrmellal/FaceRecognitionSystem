#!/usr/bin/env python3
"""
Photo Similarity Diagnostic Tool
Identifies which specific photos are causing similarity issues
"""

import numpy as np
import json
from pathlib import Path
from database import EnhancedDatabaseManager


def diagnose_similarity_issues():
    """Find exactly which photos are too similar"""

    print("🔍 PHOTO SIMILARITY DIAGNOSTIC")
    print("=" * 50)

    db = EnhancedDatabaseManager()

    # I will focus on mounia_amrhar since that's the problematic person
    person_id = "mounia_amrhar"

    if person_id not in db.known_faces:
        print(f"❌ {person_id} not found in database")
        return

    encodings = db.known_faces[person_id]
    metadata = db.encoding_validation.get(person_id, [])

    print(f"\n👤 Analyzing {person_id}:")
    print(f"📊 Total encodings: {len(encodings)}")
    print(f"📁 Available metadata: {len(metadata)}")

    # List all photos with their indices
    print(f"\n📷 Photos in database:")
    for i, meta in enumerate(metadata):
        source_image = Path(meta.get('source_image', 'unknown')).name
        quality = meta.get('quality_score', 0)
        print(f"   {i}: {source_image} (quality: {quality:.3f})")

    print(f"\n🔍 SIMILARITY ANALYSIS:")
    print("=" * 60)

    # Compare all pairs and find problematic ones
    problematic_pairs = []

    for i in range(len(encodings)):
        for j in range(i + 1, len(encodings)):
            similarity = np.dot(encodings[i], encodings[j])

            # Get image names
            img1_name = Path(metadata[i].get('source_image', f'Image_{i}')).name if i < len(metadata) else f'Image_{i}'
            img2_name = Path(metadata[j].get('source_image', f'Image_{j}')).name if j < len(metadata) else f'Image_{j}'

            # Color code by similarity level
            if similarity > 0.95:
                status = "🚨 CRITICAL"
                problematic_pairs.append((i, j, similarity, img1_name, img2_name))
            elif similarity > 0.90:
                status = "⚠️ HIGH"
            elif similarity > 0.80:
                status = "🟡 MEDIUM"
            else:
                status = "✅ GOOD"

            print(f"   {img1_name[:25]:<25} vs {img2_name[:25]:<25} = {similarity:.3f} {status}")

    # Report problematic pairs
    if problematic_pairs:
        print(f"\n🚨 PROBLEMATIC PAIRS (>95% similar):")
        print("=" * 60)

        for i, j, similarity, img1, img2 in problematic_pairs:
            print(f"   📸 {img1}")
            print(f"   📸 {img2}")
            print(f"   🔗 Similarity: {similarity:.3f} ({similarity * 100:.1f}%)")
            print(f"   💡 Action: Remove one of these photos")
            print()

        # Provide specific removal commands
        print("🛠️ RECOMMENDED ACTIONS:")
        print("=" * 30)

        for i, (idx1, idx2, sim, img1, img2) in enumerate(problematic_pairs):
            # Suggest removing the one with lower quality or simpler name
            meta1 = metadata[idx1] if idx1 < len(metadata) else {}
            meta2 = metadata[idx2] if idx2 < len(metadata) else {}

            quality1 = meta1.get('quality_score', 0)
            quality2 = meta2.get('quality_score', 0)

            if quality1 < quality2:
                remove_img = img1
                keep_img = img2
                reason = f"lower quality ({quality1:.3f} vs {quality2:.3f})"
            elif quality2 < quality1:
                remove_img = img2
                keep_img = img1
                reason = f"lower quality ({quality2:.3f} vs {quality1:.3f})"
            else:
                # Same quality, remove the one with more generic name
                if "copy" in img1.lower() or len(img1) > len(img2):
                    remove_img = img1
                    keep_img = img2
                    reason = "more generic filename"
                else:
                    remove_img = img2
                    keep_img = img1
                    reason = "more generic filename"

            print(f"Pair {i + 1}:")
            print(f"   🗑️  Remove: {remove_img} ({reason})")
            print(f"   ✅ Keep:   {keep_img}")
            print(f"   Command: rm \"known_faces/mounia_amrhar/{remove_img}\"")
            print()

    else:
        print(f"\n No problematic pairs found!")
        print("All photos have good diversity.")


if __name__ == "__main__":
    diagnose_similarity_issues()