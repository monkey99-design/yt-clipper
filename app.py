# YT Clipper (desktop): yt-dlp + Whisper lokal + OpenCV (ikuti pembicara) + ffmpeg
import os, sys, json, re, shutil, subprocess, threading, queue, urllib.request
from pathlib import Path
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont

APP = Path(sys.executable).parent if getattr(sys, 'frozen', False) else Path(__file__).parent
RES = Path(getattr(sys, '_MEIPASS', APP))
CFG = Path.home() / '.yt_clipper.json'
fmt = lambda s: f"{int(s//60)}:{int(s%60):02d}"
safe = lambda s: re.sub(r'[\\/:*?"<>|]+', '_', s).strip()[:60] or 'video'

def ffmpeg():
    d = Path.home() / '.yt_clipper' / 'bin'; d.mkdir(parents=True, exist_ok=True)
    f = d / ('ffmpeg.exe' if os.name == 'nt' else 'ffmpeg')
    if not f.exists():
        import imageio_ffmpeg; shutil.copy(imageio_ffmpeg.get_ffmpeg_exe(), f)
    return str(f)

# ---------- 1. unduh ----------
def download(url, work, log):
    import yt_dlp
    def hook(d):
        if d['status'] == 'downloading' and d.get('_percent_str'): log(f"  unduh {d['_percent_str'].strip()}", end=True)
    opts = {'format': 'bv*[height<=1080]+ba/b[height<=1080]/b', 'merge_output_format': 'mp4',
            'outtmpl': str(work / 'src.%(ext)s'), 'ffmpeg_location': str(Path(ffmpeg()).parent),
            'noplaylist': True, 'quiet': True, 'no_warnings': True, 'progress_hooks': [hook]}
    with yt_dlp.YoutubeDL(opts) as y:
        info = y.extract_info(url, download=True)
    return next(work.glob('src.*')), info

# ---------- 2. Whisper lokal ----------
def transcribe(src, work, model, lang, log):
    cache = work / f'transcript_{model}_{lang}.json'
    if cache.exists(): log('Transkrip dari cache.'); return json.loads(cache.read_text('utf-8'))
    from faster_whisper import WhisperModel
    log(f'Memuat Whisper "{model}" (pertama kali akan mengunduh model)...')
    m = WhisperModel(model, device='cpu', compute_type='int8')
    segs, _ = m.transcribe(str(src), language=None if lang == 'auto' else lang, vad_filter=True, word_timestamps=True)
    out = []
    for s in segs:
        out.append({'s': s.start, 'e': s.end, 't': s.text.strip(),
                    'w': [(w.start, w.end, w.word.strip()) for w in (s.words or [])]})
        log(f'  transkrip sampai {fmt(s.end)}', end=True)
    cache.write_text(json.dumps(out, ensure_ascii=False), 'utf-8')
    return out

def chunks(segs, start, end, maxw=5):
    ws = [w for s in segs for w in s['w'] if w[1] > start and w[0] < end]
    out, cur = [], []
    for w in ws:
        cur.append(w)
        if len(cur) >= maxw or re.search(r'[.!?,]$', w[2]) or cur[-1][1] - cur[0][0] > 2.5:
            out.append(cur); cur = []
    if cur: out.append(cur)
    return [(max(c[0][0], start) - start, min(c[-1][1], end) - start, ' '.join(x[2] for x in c)) for c in out]

def snap(segs, s, e, D, L):
    st = [x['s'] for x in segs if x['s'] <= s + 1]; en = [x['e'] for x in segs if x['e'] >= e - 1]
    s2 = max(st) if st else s; e2 = min(en) if en else e
    if e2 - s2 > L * 1.4 or e2 - s2 < L * 0.5: s2, e2 = s, e
    return max(0, s2), min(D, e2)

# ---------- 3. pilih momen ----------
def heat_clips(info, n, L):
    D = info['duration']; h = info.get('heatmap') or []; pick = []
    for p in sorted(h, key=lambda x: -x['value']):
        t = (p['start_time'] + p['end_time']) / 2
        if all(abs(t - q) >= L for q in pick): pick.append(t)
        if len(pick) == n: break
    if not pick: pick = [D * (k + .5) / n for k in range(n)]
    return sorted((max(0, min(D - L, t - L / 2)), min(D, max(0, min(D - L, t - L / 2)) + L), '') for t in pick)

def llm(provider, key, prompt):
    if provider == 'Claude':
        r = urllib.request.Request('https://api.anthropic.com/v1/messages', json.dumps(
            {'model': 'claude-sonnet-5-5', 'max_tokens': 1500, 'messages': [{'role': 'user', 'content': prompt}]}).encode(),
            {'content-type': 'application/json', 'x-api-key': key, 'anthropic-version': '2023-06-01'})
        return ''.join(b.get('text', '') for b in json.load(urllib.request.urlopen(r, timeout=180))['content'])
    r = urllib.request.Request('https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-latest:generateContent',
        json.dumps({'contents': [{'parts': [{'text': prompt}]}]}).encode(),
        {'content-type': 'application/json', 'x-goog-api-key': key})
    return ''.join(p.get('text', '') for p in json.load(urllib.request.urlopen(r, timeout=180))['candidates'][0]['content']['parts'])

def ai_clips(segs, D, n, L, provider, key):
    tr = '\n'.join(f"[{int(s['s'])}] {s['t']}" for s in segs)[:120000]
    txt = llm(provider, key, f'Berikut transkrip video dengan timestamp detik. Pilih {n} momen paling penting/menarik/berpotensi viral, '
        f'tiap klip sekitar {L} detik, mulai dan berakhir di batas kalimat, tidak tumpang tindih. '
        f'Balas HANYA JSON array: [{{"start":detik,"end":detik,"title":"judul singkat"}}]\n\n{tr}')
    arr = json.loads(re.search(r'\[.*\]', txt, re.S).group(0)); out = []
    for a in arr:
        s = max(0, float(a['start'])); e = min(D, float(a['end']))
        if e - s > L * 1.4: e = s + L * 1.4
        if e - s > 3: out.append((s, e, a.get('title', '')))
    return sorted(out)

# ---------- 4. wajah & kamera ----------
class Faces:
    def __init__(s):
        m = RES / 'models' / 'face_detection_yunet_2023mar.onnx'
        s.yn = cv2.FaceDetectorYN.create(str(m), '', (320, 320)) if m.exists() and hasattr(cv2, 'FaceDetectorYN') else None
        s.haar = None if s.yn else cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    def detect(s, img):
        w = img.shape[1]; k = min(1.0, 640 / w); sm = cv2.resize(img, None, fx=k, fy=k) if k < 1 else img
        if s.yn:
            s.yn.setInputSize((sm.shape[1], sm.shape[0])); _, f = s.yn.detect(sm)
            return [] if f is None else [tuple(float(v) / k for v in r[:4]) for r in f]
        g = cv2.cvtColor(sm, cv2.COLOR_BGR2GRAY)
        return [tuple(float(v) / k for v in b) for b in s.haar.detectMultiScale(g, 1.1, 5, minSize=(24, 24))]

def analyze(seg, faces, fps):
    """Lacak pembicara (~5x/detik): wajah dgn gerakan terbanyak, dgn histeresis. Juga cari frame thumbnail terbaik."""
    cap = cv2.VideoCapture(str(seg)); step = max(1, round(fps / 5)); prev = None
    target = .5; samples = []; best = (-1, None, .5); i = 0
    while cap.grab():
        if i % step == 0:
            _, f = cap.retrieve(); H, W = f.shape[:2]
            g = cv2.cvtColor(cv2.resize(f, (320, int(320 * H / W))), cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(g, prev) if prev is not None else np.zeros_like(g); prev = g
            sharp = cv2.Laplacian(g, cv2.CV_64F).var(); sc = 320 / W; score = sharp * .2
            fs = faces.detect(f)
            if fs:
                cand = []
                for x, y, w, h in fs:
                    r = diff[int(y * sc):int((y + h) * sc), int(x * sc):int((x + w) * sc)]
                    cand.append(((x + w / 2) / W, float(r.mean()) if r.size else 0., w * h / (W * H)))
                cur = min(cand, key=lambda c: abs(c[0] - target)); top = max(cand, key=lambda c: c[1])
                nt = top[0] if (abs(cur[0] - target) > .15 or top[1] > cur[1] * 1.5) else cur[0]
                if abs(nt - target) > .03: target = nt
                score = sharp * (.2 + cur[2] * 10)
            samples.append((i, target))
            if score > best[0]: best = (score, f.copy(), target)
        i += 1
    cap.release()
    return samples, i, best

def cam_path(samples, nf, fps):
    if not samples: return np.full(nf, .5)
    tgt = np.interp(np.arange(nf), [s[0] for s in samples], [s[1] for s in samples])
    cx, v, dt, out = tgt[0], 0., 1 / fps, np.empty(nf)
    for i, t in enumerate(tgt):               # pegas teredam kritis -> gerak kamera halus
        v += (20 * (t - cx) - 8.9 * v) * dt; cx += v * dt; out[i] = cx
    return out

# ---------- 5. subtitle & render ----------
def font(size):
    for n in ('arialbd.ttf', 'Arial Bold.ttf', 'DejaVuSans-Bold.ttf', 'seguibl.ttf'):
        try: return ImageFont.truetype(n, size)
        except OSError: pass
    return ImageFont.load_default()

def overlay(text, OW, OH, vert):
    size = int(OW * .062) if vert else int(OH * .055); f = font(size)
    d = ImageDraw.Draw(Image.new('RGBA', (1, 1))); lines, cur = [], ''
    for w in text.split():
        if d.textlength((cur + ' ' + w).strip(), font=f) > OW * .88 and cur: lines.append(cur); cur = w
        else: cur = (cur + ' ' + w).strip()
    lines.append(cur); lh = int(size * 1.25); sw = max(2, size // 9)
    im = Image.new('RGBA', (OW, lh * len(lines) + 2 * sw), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    for k, l in enumerate(lines):
        d.text((OW / 2, sw + k * lh), l, font=f, fill='white', stroke_width=sw, stroke_fill='black', anchor='ma')
    a = np.asarray(im, np.float32); y0 = int(OH * (.74 if vert else .88)) - im.height // 2
    y0 = max(0, min(OH - im.height, y0))
    return a[..., :3][..., ::-1], a[..., 3:] / 255., y0

def blit(frame, ov):
    rgb, al, y0 = ov; h = rgb.shape[0]; reg = frame[y0:y0 + h]
    reg[:] = (reg * (1 - al) + rgb * al).astype(np.uint8)

def crop_resize(f, x, cw, OW, OH, vert):
    if vert: f = f[:, x:x + cw]
    return cv2.resize(f, (OW, OH), interpolation=cv2.INTER_CUBIC)

def srt_time(t): return f"{int(t//3600):02d}:{int(t%3600//60):02d}:{int(t%60):02d},{int(t%1*1000):03d}"

def render(src, s, e, title, segs, mode, faces, outdir, idx, work, log):
    ff = ffmpeg(); seg = work / 'seg.mp4'; vert = mode != 'wide'; OW, OH = (1080, 1920) if vert else (1920, 1080)
    subprocess.run([ff, '-y', '-ss', f'{s:.3f}', '-i', str(src), '-t', f'{e - s:.3f}', '-c:v', 'libx264', '-preset', 'ultrafast',
                    '-crf', '16', '-c:a', 'aac', '-b:a', '192k', str(seg)], check=True, capture_output=True)
    cap = cv2.VideoCapture(str(seg)); W = int(cap.get(3)); H = int(cap.get(4)); fps = cap.get(5) or 30; cap.release()
    cw = int(H * 9 / 16) // 2 * 2
    samples, nf, best = analyze(seg, faces, fps)
    path = cam_path(samples, nf, fps) if mode == 'follow' else np.full(nf, .5)
    xo = lambda p: int(max(0, min(W - cw, p * W - cw / 2)))
    subs = chunks(segs, s, e); ovs = [overlay(t, OW, OH, vert) for _, _, t in subs]
    name = f'klip-{idx:02d}'
    (outdir / f'{name}.srt').write_text('\n'.join(f"{k+1}\n{srt_time(a)} --> {srt_time(b)}\n{t}\n" for k, (a, b, t) in enumerate(subs)), 'utf-8')
    p = subprocess.Popen([ff, '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{OW}x{OH}', '-r', f'{fps}', '-i', '-', '-i', str(seg),
        '-map', '0:v', '-map', '1:a?', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '20', '-pix_fmt', 'yuv420p',
        '-c:a', 'aac', '-shortest', '-movflags', '+faststart', str(outdir / f'{name}.mp4')], stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
    cap = cv2.VideoCapture(str(seg)); i = 0
    while True:
        ok, f = cap.read()
        if not ok: break
        fr = crop_resize(f, xo(path[min(i, nf - 1)]), cw, OW, OH, vert); t = i / fps
        for (a, b, _), ov in zip(subs, ovs):
            if a <= t < b: blit(fr, ov); break
        p.stdin.write(fr.tobytes()); i += 1
    cap.release(); p.stdin.close(); p.wait()
    if best[1] is not None:                                   # thumbnail
        th = crop_resize(best[1], xo(best[2] if mode == 'follow' else .5), cw, OW, OH, vert)
        txt = title or ' '.join(' '.join(c[2] for c in subs[:2]).split()[:6])
        if txt: blit(th, overlay(txt, OW, OH, vert))
        ok, buf = cv2.imencode('.jpg', th, [cv2.IMWRITE_JPEG_QUALITY, 92]); (outdir / f'{name}-thumb.jpg').write_bytes(buf.tobytes())
    log(f'  ✔ {name}.mp4 / .srt / -thumb.jpg  ({fmt(s)}–{fmt(e)})')

# ---------- pipeline ----------
def run(c, log):
    base = Path(c['out']); base.mkdir(parents=True, exist_ok=True)
    tmpdir = base / '.work'; tmpdir.mkdir(exist_ok=True)
    log('Mengunduh video...'); src, info = download(c['url'], tmpdir, log)
    out = base / safe(info.get('title', 'video')); out.mkdir(exist_ok=True)
    segs = transcribe(src, tmpdir, c['whisper'], c['lang'], log); D = info.get('duration') or segs[-1]['e']
    log('Memilih momen terbaik...')
    if c['pick'] == 'ai' and c['key']: clips = ai_clips(segs, D, c['n'], c['len'], c['provider'], c['key'])
    else:
        if c['pick'] == 'ai': log('API key kosong, pakai heatmap.')
        clips = [(*snap(segs, s, e, D, c['len']), t) for s, e, t in heat_clips(info, c['n'], c['len'])]
    log(f'{len(clips)} klip dipilih: ' + ', '.join(f'{fmt(a)}-{fmt(b)}' for a, b, _ in clips))
    faces = Faces(); log('Deteksi wajah: ' + ('YuNet' if faces.yn else 'Haar (model YuNet tidak ada)'))
    for k, (s, e, t) in enumerate(clips, 1):
        log(f'Render klip {k}/{len(clips)}...'); render(src, s, e, t, segs, c['mode'], faces, out, k, tmpdir, log)
    log(f'SELESAI. Folder: {out}')

# ---------- GUI ----------
def main():
    import tkinter as tk
    from tkinter import ttk, filedialog
    cfg = json.loads(CFG.read_text()) if CFG.exists() else {}
    root = tk.Tk(); root.title('YT Clipper'); root.geometry('640x620')
    v = {k: tk.StringVar(value=cfg.get(k, d)) for k, d in dict(url='', out=str(Path.home() / 'Videos' / 'YT Clipper'),
         mode='Vertikal 9:16 – ikuti pembicara', pick='Heatmap YouTube', provider='Claude', key='', whisper='base', lang='auto', n='5', len='30').items()}
    f = ttk.Frame(root, padding=10); f.pack(fill='x')
    def row(label, var, vals=None, show=None):
        r = ttk.Frame(f); r.pack(fill='x', pady=2); ttk.Label(r, text=label, width=18).pack(side='left')
        w = ttk.Combobox(r, textvariable=var, values=vals, state='readonly') if vals else ttk.Entry(r, textvariable=var, show=show or '')
        w.pack(side='left', fill='x', expand=True); return r
    row('URL YouTube', v['url'])
    r = row('Folder hasil', v['out']); ttk.Button(r, text='...', width=3, command=lambda: v['out'].set(filedialog.askdirectory() or v['out'].get())).pack(side='left')
    row('Rasio', v['mode'], ['Vertikal 9:16 – ikuti pembicara', 'Vertikal 9:16 – tengah', 'Asli 16:9'])
    row('Pilih momen', v['pick'], ['Heatmap YouTube', 'AI dari transkrip'])
    row('Provider AI', v['provider'], ['Claude', 'Gemini']); row('API key (opsional)', v['key'], show='*')
    row('Model Whisper', v['whisper'], ['tiny', 'base', 'small', 'medium']); row('Bahasa', v['lang'], ['auto', 'id', 'en'])
    row('Jumlah klip (3-10)', v['n']); row('Durasi klip (detik)', v['len'])
    q = queue.Queue(); box = tk.Text(root, height=18, state='disabled'); box.pack(fill='both', expand=True, padx=10, pady=6)
    def log(m, end=False): q.put((m, end))
    def pump():
        while not q.empty():
            m, end = q.get(); box.config(state='normal')
            if end: box.delete('end-2l', 'end-1l')
            box.insert('end', m + '\n'); box.see('end'); box.config(state='disabled')
        root.after(150, pump)
    def go():
        c = {k: x.get().strip() for k, x in v.items()}; CFG.write_text(json.dumps({k: x.get() for k, x in v.items()}))
        c['mode'] = {'Vertikal 9:16 – ikuti pembicara': 'follow', 'Vertikal 9:16 – tengah': 'center'}.get(c['mode'], 'wide')
        c['pick'] = 'ai' if c['pick'].startswith('AI') else 'heat'
        c['n'] = max(3, min(10, int(c['n'] or 5))); c['len'] = max(10, min(120, int(c['len'] or 30)))
        btn.config(state='disabled')
        def t():
            try: run(c, log)
            except Exception as e: log(f'GAGAL: {e}')
            root.after(0, lambda: btn.config(state='normal'))
        threading.Thread(target=t, daemon=True).start()
    btn = ttk.Button(root, text='Mulai', command=go); btn.pack(pady=6); pump(); root.mainloop()

if __name__ == '__main__': main()
