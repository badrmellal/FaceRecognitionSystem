#!/usr/bin/env python3
"""
Database Encoding Fix
Fixes the normalization and validation issues in face encodings
"""

import numpy as np
import json
import pickle
from pathlib import Path
from database import EnhancedDatabaseManager


def validate_and_fix_encodings():
    """Validate and fix encoding issues"""

    db = EnhancedDatabaseManager()

    print("🔍 DIAGNOSING ENCODING ISSUES")
    print("=" * 50)

    fixed_encodings = {}
    issues_found = []

    for person_id, encodings in db.known_faces.items():
        print(f"\n👤 Analyzing {person_id}:")

        valid_encodings = []
        person_issues = []

        for i, encoding in enumerate(encodings):
            enc_array = np.array(encoding)

            # Check 1: Proper shape
            if enc_array.shape != (512,):
                person_issues.append(f"  ❌ Encoding {i}: Wrong shape {enc_array.shape}")
                continue

            # Check 2: No NaN or infinity
            if not np.all(np.isfinite(enc_array)):
                person_issues.append(f"  ❌ Encoding {i}: Contains NaN/Inf")
                continue

            # Check 3: Not all zeros
            if np.all(enc_array == 0):
                person_issues.append(f"  ❌ Encoding {i}: All zeros")
                continue

            # Check 4: Proper normalization (should be close to 1.0)
            norm = np.linalg.norm(enc_array)
            if norm < 0.5 or norm > 2.0:
                person_issues.append(f"  ⚠️ Encoding {i}: Abnormal norm {norm:.3f}")

            # Fix normalization
            try:
                normalized_enc = enc_array / (norm + 1e-8)

                # Verify the fix
                new_norm = np.linalg.norm(normalized_enc)
                if 0.98 <= new_norm <= 1.02:  # Should be very close to 1.0
                    valid_encodings.append(normalized_enc.tolist())
                    print(f"  ✅ Encoding {i}: Fixed norm {norm:.3f} → {new_norm:.3f}")
                else:
                    person_issues.append(f"  ❌ Encoding {i}: Cannot fix norm {norm:.3f}")

            except Exception as e:
                person_issues.append(f"  ❌ Encoding {i}: Fix failed - {e}")

        if valid_encodings:
            # Calculate internal similarities for validation
            similarities = []
            for i in range(len(valid_encodings)):
                for j in range(i + 1, len(valid_encodings)):
                    sim = np.dot(valid_encodings[i], valid_encodings[j])
                    similarities.append(sim)

            if similarities:
                min_sim = min(similarities)
                max_sim = max(similarities)
                avg_sim = np.mean(similarities)

                print(f"  📊 Internal similarities: {min_sim:.3f} to {max_sim:.3f} (avg: {avg_sim:.3f})")

                # Flag suspicious similarities
                if min_sim < 0:
                    person_issues.append(f"  🚨 NEGATIVE similarity detected: {min_sim:.3f}")
                if max_sim > 0.95:
                    person_issues.append(f"  🚨 EXCESSIVE similarity detected: {max_sim:.3f}")

                # Only keep if similarities are reasonable
                if min_sim >= 0 and max_sim <= 0.95:
                    fixed_encodings[person_id] = valid_encodings
                    print(f"  ✅ {person_id}: {len(valid_encodings)} valid encodings")
                else:
                    print(f"  ❌ {person_id}: Rejected due to suspicious similarities")
            else:
                fixed_encodings[person_id] = valid_encodings
                print(f"  ✅ {person_id}: {len(valid_encodings)} encodings (single encoding)")

        if person_issues:
            issues_found.extend([f"{person_id}:"] + person_issues)

    # Report issues
    if issues_found:
        print(f"\n🚨 ISSUES FOUND:")
        for issue in issues_found:
            print(issue)

    # Save fixed database
    if fixed_encodings:
        print(f"\n💾 SAVING FIXED DATABASE:")

        # Backup original
        backup_path = db.db_path / f"backup_broken_{int(time.time())}.pkl"
        if db.encodings_file.exists():
            import shutil
            shutil.copy2(db.encodings_file, backup_path)
            print(f"  📦 Original backed up to: {backup_path}")

        # Save fixed encodings
        with open(db.encodings_file, 'wb') as f:
            pickle.dump(fixed_encodings, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"  ✅ Fixed database saved with {len(fixed_encodings)} persons")

        # Update in-memory
        db.known_faces = fixed_encodings

        # Validate the fix
        print(f"\n🔬 VALIDATING FIX:")
        for person_id, encodings in fixed_encodings.items():
            enc_array = np.array(encodings)
            norms = [np.linalg.norm(enc) for enc in encodings]
            print(f"  {person_id}: {len(encodings)} encodings, norms: {min(norms):.3f}-{max(norms):.3f}")

    return len(fixed_encodings)


def test_cross_similarities():
    """Test similarities between different people"""
    print(f"\n🧪 TESTING CROSS-PERSON SIMILARITIES")
    print("=" * 50)

    db = EnhancedDatabaseManager()

    people = list(db.known_faces.keys())
    if len(people) < 2:
        print("Need at least 2 people for cross-testing")
        return

    for i, person1 in enumerate(people):
        for j, person2 in enumerate(people):
            if i >= j:
                continue

            # Get best similarity between the two people
            max_similarity = 0.0
            for enc1 in db.known_faces[person1]:
                for enc2 in db.known_faces[person2]:
                    sim = np.dot(enc1, enc2)
                    max_similarity = max(max_similarity, sim)

            status = "🚨 DANGEROUS" if max_similarity > 0.8 else "⚠️ HIGH" if max_similarity > 0.7 else "✅ SAFE"
            print(f"  {person1} vs {person2}: {max_similarity:.3f} {status}")

            if max_similarity > 0.8:
                print(f"    💡 These people are too similar - consider re-taking photos")


if __name__ == "__main__":
    import time

    print("🛠️ FACE ENCODING DATABASE REPAIR TOOL")
    print("=" * 60)

    # Step 1: Fix encodings
    fixed_count = validate_and_fix_encodings()

    if fixed_count > 0:
        # Step 2: Test cross-similarities
        test_cross_similarities()

        print(f"\n✅ DATABASE REPAIR COMPLETED")
        print(f"Fixed {fixed_count} person encodings")
        print(f"Please restart your recognition system to use the fixed database")
    else:
        print(f"\n❌ NO VALID ENCODINGS FOUND")
        print(f"You may need to regenerate the database with better quality images")