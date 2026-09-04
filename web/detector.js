/* ============================================================
   Pipeline de detection, sans dependance au DOM de la page.

   Le modele est exporte sans NMS (nms=False) : le decodage de la sortie brute
   et la suppression des non-maxima sont faits ici.
   ============================================================ */

const Detector = (() => {

  /** Redimensionne en conservant le ratio et complete par du gris 114,
   *  exactement comme le letterbox d'Ultralytics.
   *  @returns {{tensor: Float32Array, scale: number, dx: number, dy: number}} */
  function letterbox(ctx, source, W, H, srcW, srcH) {
    const scale = Math.min(W / srcW, H / srcH);
    const nw = Math.round(srcW * scale);
    const nh = Math.round(srcH * scale);
    const dx = Math.floor((W - nw) / 2);
    const dy = Math.floor((H - nh) / 2);

    ctx.fillStyle = '#727272';
    ctx.fillRect(0, 0, W, H);
    ctx.drawImage(source, dx, dy, nw, nh);

    const { data } = ctx.getImageData(0, 0, W, H);
    const area = W * H;
    const tensor = new Float32Array(area * 3);
    for (let i = 0; i < area; i++) {
      tensor[i] = data[i * 4] / 255;                  // R
      tensor[area + i] = data[i * 4 + 1] / 255;       // G
      tensor[area * 2 + i] = data[i * 4 + 2] / 255;   // B
    }
    return { tensor, scale, dx, dy };
  }

  /** Sortie brute (1, 4+nc, N) : cx, cy, w, h puis un score par classe.
   *  Les boites reviennent dans le repere de l'image d'origine. */
  function decode(output, geom, confThr) {
    const [, nAttr, nBox] = output.dims;
    const data = output.data;
    const nc = nAttr - 4;
    const boxes = [];

    for (let i = 0; i < nBox; i++) {
      let best = 0;
      let bestScore = data[4 * nBox + i];
      for (let c = 1; c < nc; c++) {
        const s = data[(4 + c) * nBox + i];
        if (s > bestScore) { bestScore = s; best = c; }
      }
      if (bestScore < confThr) continue;

      const cx = data[i];
      const cy = data[nBox + i];
      const w = data[2 * nBox + i];
      const h = data[3 * nBox + i];

      boxes.push({
        cls: best,
        score: bestScore,
        x: (cx - w / 2 - geom.dx) / geom.scale,
        y: (cy - h / 2 - geom.dy) / geom.scale,
        w: w / geom.scale,
        h: h / geom.scale,
      });
    }
    return { boxes, nc };
  }

  function iou(a, b) {
    const x1 = Math.max(a.x, b.x);
    const y1 = Math.max(a.y, b.y);
    const x2 = Math.min(a.x + a.w, b.x + b.w);
    const y2 = Math.min(a.y + a.h, b.y + b.h);
    const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
    if (inter <= 0) return 0;
    return inter / (a.w * a.h + b.w * b.h - inter);
  }

  /** NMS par classe : deux vehicules distincts peuvent se recouvrir fortement,
   *  et une voiture ne doit jamais eliminer sa propre plaque. */
  function nms(boxes, thr, nc) {
    const kept = [];
    for (let c = 0; c < nc; c++) {
      const pool = boxes.filter((b) => b.cls === c).sort((a, b) => b.score - a.score);
      while (pool.length) {
        const best = pool.shift();
        kept.push(best);
        for (let i = pool.length - 1; i >= 0; i--) {
          if (iou(best, pool[i]) > thr) pool.splice(i, 1);
        }
      }
    }
    return kept.sort((a, b) => b.score - a.score);
  }

  /** Chaine complete : pretraitement, inference, decodage, NMS. */
  async function run(session, ctx, source, W, H, srcW, srcH, opts) {
    const geom = letterbox(ctx, source, W, H, srcW, srcH);
    const feeds = {};
    feeds[session.inputNames[0]] = new ort.Tensor('float32', geom.tensor, [1, 3, H, W]);
    const out = await session.run(feeds);
    const tensor = out[session.outputNames[0]];
    const { boxes, nc } = decode(tensor, geom, opts.conf);
    return { detections: nms(boxes, opts.iou, nc), dims: tensor.dims, nc };
  }

  return { letterbox, decode, iou, nms, run };
})();

if (typeof module !== 'undefined') module.exports = Detector;
