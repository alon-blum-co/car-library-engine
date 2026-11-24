# -*- coding: utf-8 -*-
"""Background removal for car photos (Wikimedia Commons, free licenses).
Input: jobs/<batch>.json = [{"id": "...", "url": "..."}]. Output: out/<id>.png (RGBA, cropped) + out/report.json.
Keeps only the main car: largest connected region of the mask, plus regions touching it."""
import json, sys, os, time, io, urllib.request
import numpy as np
from PIL import Image
from rembg import new_session, remove
from scipy import ndimage

job = json.load(open(sys.argv[1]))
if len(sys.argv) > 3:   # chunk i of n (parallel matrix)
    i, n = int(sys.argv[2]), int(sys.argv[3]); job = job[i::n]
os.makedirs('out', exist_ok=True)
sess = new_session(os.environ.get('MODEL', 'birefnet-general'))
try:
    from open_image_models import LicensePlateDetector
    plate_det = LicensePlateDetector(detection_model='yolo-v9-s-608-license-plate-end2end')
except Exception as e:
    print('plate detector unavailable', e); plate_det = None
UA = {'User-Agent': 'car-library-engine/0.1 (open data research)'}
rep = []
for it in job:
    t0 = time.time(); r = {'id': it['id']}
    try:
        b = urllib.request.urlopen(urllib.request.Request(it['url'], headers=UA), timeout=90).read()
        im = Image.open(io.BytesIO(b)).convert('RGB')
        plates = []
        if plate_det is not None:
            import cv2
            for p in plate_det.predict(cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)):
                bb = p.bounding_box
                plates.append({'box': [int(bb.x1), int(bb.y1), int(bb.x2), int(bb.y2)], 'conf': round(float(p.confidence), 3)})
        cut = remove(im, session=sess, post_process_mask=True)
        a = np.asarray(cut)[..., 3]
        lab, n = ndimage.label(a > 60)
        if n > 1:
            sizes = ndimage.sum(np.ones_like(a), lab, range(1, n + 1))
            keep = np.argmax(sizes) + 1
            m = ndimage.binary_dilation(lab == keep, iterations=3) & (a > 0)
            a = np.where(m, a, 0).astype(np.uint8)
            cut.putalpha(Image.fromarray(a))
        ys, xs = np.nonzero(a > 20); H, W = a.shape
        x0, y0 = int(xs.min()), int(ys.min())
        cut = cut.crop((x0, y0, xs.max() + 1, ys.max() + 1))
        # לוחיות בקואורדינטות של התמונה החתוכה, רק כאלה שיושבות על הרכב שנשאר
        r['plates'] = [{'box': [p['box'][0] - x0, p['box'][1] - y0, p['box'][2] - x0, p['box'][3] - y0], 'conf': p['conf']}
                       for p in plates if a[min(H - 1, (p['box'][1] + p['box'][3]) // 2), min(W - 1, (p['box'][0] + p['box'][2]) // 2)] > 128]
        r.update(ok=True, w=cut.width, h=cut.height, src_w=W, src_h=H,
                 area=round(cut.width * cut.height / (W * H), 3),
                 touches=[bool(xs.min() <= 2), bool(ys.min() <= 2), bool(xs.max() >= W - 3), bool(ys.max() >= H - 3)],
                 fill=round(float((a > 128).sum()) / (cut.width * cut.height), 3), parts=int(n), sec=round(time.time() - t0, 1))
        cut.save(f'out/{it["id"]}.png', optimize=True)
    except Exception as e:
        r.update(ok=False, err=str(e)[:200])
    rep.append(r); print(r, flush=True); time.sleep(1)
json.dump(rep, open('out/report.json', 'w'), indent=1)
