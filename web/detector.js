// Detection pipeline, independent of the page DOM.
// The model is exported without NMS, so decoding the raw output and running
// non-maximum suppression happen here.

const Detector = (() => {

  // Resize keeping the aspect ratio and pad with grey 114, exactly like the
  // Ultralytics letterbox. Returns the tensor and what undoes the transform.
  function letterbox(ctx, source) {
    const { width: W, height: H } = ctx.canvas;
    const scale = Math.min(W / source.width, H / source.height);
    const nw = Math.round(source.width * scale);
    const nh = Math.round(source.height * scale);
    const dx = Math.floor((W - nw) / 2);
    const dy = Math.floor((H - nh) / 2);

    ctx.fillStyle = '#727272';
    ctx.fillRect(0, 0, W, H);
    ctx.drawImage(source, dx, dy, nw, nh);

    const { data } = ctx.getImageData(0, 0, W, H);
    const area = W * H;
    const tensor = new Float32Array(area * 3);
    for (let i = 0; i < area; i++) {
      tensor[i] = data[i * 4] / 255;
      tensor[area + i] = data[i * 4 + 1] / 255;
      tensor[area * 2 + i] = data[i * 4 + 2] / 255;
    }
    return { tensor, scale, dx, dy };
  }

  // Output is (1, 4+nc, N) with the boxes on the fastest axis, so attribute `a`
  // of box `i` sits at a * N + i. Boxes come back in the source image's space.
  function decode(output, geom, confThr) {
    const [, nAttr, nBox] = output.dims;
    const data = output.data;
    const at = (a, i) => data[a * nBox + i];
    const nc = nAttr - 4;
    const boxes = [];

    for (let i = 0; i < nBox; i++) {
      let cls = 0;
      let score = at(4, i);
      for (let c = 1; c < nc; c++) {
        const s = at(4 + c, i);
        if (s > score) { score = s; cls = c; }
      }
      if (score < confThr) continue;

      const cx = at(0, i), cy = at(1, i), w = at(2, i), h = at(3, i);
      boxes.push({
        cls,
        score,
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
    return inter > 0 ? inter / (a.w * a.h + b.w * b.h - inter) : 0;
  }

  // Per class, never globally: a car must not suppress its own number plate.
  function nms(boxes, thr, nc) {
    const kept = [];
    for (let c = 0; c < nc; c++) {
      let pool = boxes.filter((b) => b.cls === c).sort((a, b) => b.score - a.score);
      while (pool.length) {
        const best = pool.shift();
        kept.push(best);
        pool = pool.filter((b) => iou(best, b) <= thr);
      }
    }
    return kept.sort((a, b) => b.score - a.score);
  }

  async function run(session, ctx, source, opts) {
    const { width: W, height: H } = ctx.canvas;
    const geom = letterbox(ctx, source);
    const feeds = {
      [session.inputNames[0]]: new ort.Tensor('float32', geom.tensor, [1, 3, H, W]),
    };
    const tensor = (await session.run(feeds))[session.outputNames[0]];
    const { boxes, nc } = decode(tensor, geom, opts.conf);
    return { detections: nms(boxes, opts.iou, nc), dims: tensor.dims, nc };
  }

  return { letterbox, decode, iou, nms, run };
})();

if (typeof module !== 'undefined') module.exports = Detector;
