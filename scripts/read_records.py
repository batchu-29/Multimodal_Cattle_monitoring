"""Read the latest gait records back from Firestore and save their images.

    python scripts/read_records.py            # latest 1 record
    python scripts/read_records.py 5          # latest 5
    python scripts/read_records.py 1 Gauri    # latest 1 for one cow

Writes out/records/<record_id>/image.jpg and overlay.jpg and prints the rest of the fields.
Read-only: it never changes or deletes anything.
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import gait_detection as gd  # noqa: E402  (also loads .env)

if not gd.firebase_configured():
    sys.exit("FIREBASE_CREDENTIALS is not set to an existing file (see .env.example).")

from firebase_admin import firestore  # noqa: E402

n = int(sys.argv[1]) if len(sys.argv) > 1 else 1
cow = sys.argv[2] if len(sys.argv) > 2 else None
client = gd.FirestoreWriter().client
collection = client.collection(gd.FIRESTORE_COLLECTION)
if cow:
    # filter + order in one query would need a composite index, so sort here (one cow = few records)
    docs = sorted(collection.where("cow_name", "==", cow).stream(),
                  key=lambda d: d.to_dict()["recorded_at"], reverse=True)[:n]
else:
    docs = list(collection.order_by("recorded_at", direction=firestore.Query.DESCENDING).limit(n).stream())
if not docs:
    sys.exit("no records found")

for doc in docs:
    data = doc.to_dict()
    out_dir = os.path.join(ROOT, "out", "records", doc.id)
    os.makedirs(out_dir, exist_ok=True)
    for field, name in (("image", "image.jpg"), ("overlay_image", "overlay.jpg")):
        blob = data.get(field)
        if blob:
            with open(os.path.join(out_dir, name), "wb") as fh:
                fh.write(blob)
    shown = {k: v for k, v in data.items() if k not in ("image", "overlay_image", "keypoints", "frames")}
    shown["recorded_at"] = str(shown.get("recorded_at"))
    shown["image_bytes"] = len(data.get("image") or b"")
    shown["overlay_bytes"] = len(data.get("overlay_image") or b"")
    shown["keypoints"] = f"{len(data.get('keypoints') or {})} points"
    shown["frames"] = [f"t={f['time_s']}s rank {f['rank']}: {len(f['keypoints'])} points" for f in data.get("frames", [])]
    print(f"== {doc.id}  (images saved to {out_dir})")
    print(json.dumps(shown, indent=2, default=str))
