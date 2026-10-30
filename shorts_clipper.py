# -*- coding: utf-8 -*-
"""
AI Shorts Clipper (100% open-source stack)
------------------------------------------
YouTube लंबा वीडियो (URL) / अपनी फाइल  ->  डाउनलोड (yt-dlp)  ->  ट्रांसक्रिप्शन (faster-whisper, लोकल)
->  वायरल हाइलाइट खोज (Gemini / Ollama / बिना-key Heuristic)  ->  9:16 क्लिप + कैप्शन + एडिटिंग (ffmpeg)

चलाने के लिए:
    pip install -r requirements.txt
    ffmpeg इंस्टॉल करें (https://ffmpeg.org)
    streamlit run shorts_clipper.py
"""
import os
import re
import json
import math
import time
import wave
import shutil
import zipfile
import subprocess
from pathlib import Path

import numpy as np
import requests
import streamlit as st

WORK = Path("workspace")
WORK.mkdir(exist_ok=True)

st.set_page_config(page_title="AI Shorts Clipper", page_icon="✂️", layout="wide")

# ----------------------------------------------------------------------------
# छोटे हेल्पर्स
# ----------------------------------------------------------------------------


def run(cmd, cwd=None):
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="ignore")
    if p.returncode != 0:
        raise RuntimeError((p.stderr or "")[-2500:])
    return p.stdout


def fmt(t):
    t = int(max(0, t))
    m, s = divmod(t, 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def safe_name(s, n=50):
    s = re.sub(r"[^\w\s-]", "", s, flags=re.UNICODE).strip()
    s = re.sub(r"\s+", "_", s)
    return (s[:n] or "clip")


def ffmpeg_ok():
    return shutil.which("ffmpeg") is not None


# ----------------------------------------------------------------------------
# 1) डाउनलोड
# ----------------------------------------------------------------------------


def download_video(url, out_dir, max_h, cookies_path=None, on_progress=None):
    import yt_dlp

    out_dir.mkdir(parents=True, exist_ok=True)

    def hook(d):
        if on_progress and d.get("status") == "downloading":
            tot = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
            if tot:
                on_progress(min(1.0, d.get("downloaded_bytes", 0) / tot))

    opts = {
        "format": f"bestvideo[height<={max_h}]+bestaudio/best[height<={max_h}]/best",
        "merge_output_format": "mp4",
        "outtmpl": str(out_dir / "source.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "progress_hooks": [hook],
    }
    if cookies_path:
        opts["cookiefile"] = cookies_path
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
    files = sorted(out_dir.glob("source.*"))
    if not files:
        raise RuntimeError("डाउनलोड फेल हो गया")
    mp4 = [f for f in files if f.suffix == ".mp4"]
    return (mp4[0] if mp4 else files[0]), info.get("title", "video")


def get_duration(path):
    out = run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
               "-of", "default=nw=1:nk=1", str(path)])
    return float(out.strip())


def extract_audio(src, wav_path):
    run(["ffmpeg", "-y", "-i", str(src), "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(wav_path)])


# ----------------------------------------------------------------------------
# 2) ट्रांसक्रिप्शन (लोकल, faster-whisper)
# ----------------------------------------------------------------------------


def transcribe(wav_path, model_size, language, duration, on_progress=None):
    from faster_whisper import WhisperModel

    model = WhisperModel(model_size, device="auto", compute_type="auto")
    seg_iter, info = model.transcribe(
        str(wav_path),
        language=(None if language == "auto" else language),
        word_timestamps=True,
        vad_filter=True,
    )
    segs = []
    for s in seg_iter:
        words = [{"w": w.word.strip(), "s": float(w.start), "e": float(w.end)}
                 for w in (s.words or []) if w.word.strip()]
        segs.append({"start": float(s.start), "end": float(s.end), "text": s.text.strip(), "words": words})
        if on_progress and duration:
            on_progress(min(1.0, s.end / duration))
    return segs, info.language


# ----------------------------------------------------------------------------
# 3) हाइलाइट खोज
# ----------------------------------------------------------------------------


def call_gemini(key, model, prompt):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    r = requests.post(
        url,
        headers={"x-goog-api-key": key, "Content-Type": "application/json"},
        json={"contents": [{"parts": [{"text": prompt}]}],
              "generationConfig": {"responseMimeType": "application/json", "temperature": 0.4}},
        timeout=300,
    )
    r.raise_for_status()
    return r.json()["candidates"][0]["content"]["parts"][0]["text"]


def call_ollama(host, model, prompt):
    r = requests.post(
        host.rstrip("/") + "/api/chat",
        json={"model": model, "messages": [{"role": "user", "content": prompt}],
              "stream": False, "format": "json",
              "options": {"temperature": 0.4, "num_ctx": 16384}},
        timeout=1200,
    )
    r.raise_for_status()
    return r.json()["message"]["content"]


def parse_json(txt):
    txt = txt.strip()
    txt = re.sub(r"^```(?:json)?|```$", "", txt, flags=re.M).strip()
    try:
        return json.loads(txt)
    except Exception:
        a, b = txt.find("{"), txt.rfind("}")
        if a != -1 and b != -1:
            try:
                return json.loads(txt[a:b + 1])
            except Exception:
                pass
    return {"clips": []}


def make_chunks(segs, max_chars):
    chunks, cur, size = [], [], 0
    for s in segs:
        line = f"[{int(s['start'])}] {s['text']}"
        if size + len(line) > max_chars and cur:
            chunks.append("\n".join(cur))
            cur, size = [], 0
        cur.append(line)
        size += len(line) + 1
    if cur:
        chunks.append("\n".join(cur))
    return chunks


def build_prompt(chunk, k, min_d, max_d, extra):
    return f"""You are an expert short-form video editor (YouTube Shorts / Reels / TikTok).
Below is a timestamped transcript of a long video. Each line starts with the time in SECONDS.
Pick the {k} BEST moments that can go viral as standalone vertical clips.

Rules:
- Each clip must be between {min_d} and {max_d} seconds long.
- It must start with a strong hook (a surprising, funny, emotional, controversial or high-value moment) and end at a natural stopping point.
- It must make sense without the rest of the video. Clips must not overlap.
- Use start/end values in seconds based on the transcript timestamps.
- "title" must be catchy, max 70 characters, in the SAME language as the transcript.
{('- Extra instruction from the user: ' + extra) if extra else ''}

Return ONLY valid JSON in this exact shape:
{{"clips":[{{"start":12.0,"end":58.0,"title":"...","hook":"first line of the clip","score":87,"reason":"why it can go viral"}}]}}

TRANSCRIPT:
{chunk}
"""


def snap(start, end, segs, min_d, max_d):
    if segs:
        i = min(range(len(segs)), key=lambda k: abs(segs[k]["start"] - start))
        if abs(segs[i]["start"] - start) <= 6:
            start = segs[i]["start"]
    end = min(end, start + max_d)
    if segs:
        j = min(range(len(segs)), key=lambda k: abs(segs[k]["end"] - end))
        if abs(segs[j]["end"] - end) <= 6 and segs[j]["end"] > start + min_d * 0.5:
            end = segs[j]["end"]
    if end - start > max_d + 6:
        end = start + max_d
    return start, end


def overlap(a, b):
    o = max(0.0, min(a["end"], b["end"]) - max(a["start"], b["start"]))
    return o / max(1e-6, min(a["end"] - a["start"], b["end"] - b["start"]))


def pick(cands, n):
    cands = sorted(cands, key=lambda c: -c["score"])
    out = []
    for c in cands:
        if all(overlap(c, o) < 0.3 for o in out):
            out.append(c)
        if len(out) >= n:
            break
    return out


def heuristic_clips(wav_path, segs, n, min_d, max_d, duration):
    """बिना किसी API key के: आवाज़ की ऊर्जा + बोलने की रफ़्तार + !/? के आधार पर स्कोर."""
    with wave.open(str(wav_path)) as w:
        sr = w.getframerate()
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32)
    sec = int(len(data) // sr)
    if sec < 5:
        return []
    rms = np.array([np.sqrt(np.mean(data[i * sr:(i + 1) * sr] ** 2) + 1e-6) for i in range(sec)])
    z = (rms - rms.mean()) / (rms.std() + 1e-6)
    dens = np.zeros(sec)
    exc = np.zeros(sec)
    for s in segs:
        a, b = int(s["start"]), min(sec - 1, int(s["end"]))
        nw = max(1, len(s["text"].split()))
        d = max(1, b - a + 1)
        e = (s["text"].count("!") + s["text"].count("?")) / d
        for t in range(a, b + 1):
            dens[t] += nw / d
            exc[t] += e
    dens = (dens - dens.mean()) / (dens.std() + 1e-6)
    score_sec = z + 0.5 * dens + 0.8 * exc
    target = int((min_d + max_d) / 2)
    cs = np.concatenate([[0], np.cumsum(score_sec)])
    cands = []
    for st_ in range(0, max(1, sec - target), 5):
        m = (cs[min(sec, st_ + target)] - cs[st_]) / target
        cands.append((m, st_))
    cands.sort(reverse=True)
    out = []
    for m, st_ in cands:
        s0, e0 = snap(float(st_), float(st_ + target), segs, min_d, max_d)
        c = {"start": s0, "end": e0, "score": float(m)}
        if all(overlap(c, o) < 0.3 for o in out):
            txt = " ".join(x["text"] for x in segs if x["start"] >= s0 and x["end"] <= e0)
            c["title"] = " ".join(txt.split()[:8]) or "Highlight"
            c["hook"] = ""
            c["reason"] = "ऊँची आवाज़/तेज़ बोलचाल वाला हिस्सा"
            out.append(c)
        if len(out) >= n:
            break
    mx = max([c["score"] for c in out], default=1) or 1
    for c in out:
        c["score"] = int(max(1, min(100, 60 + 40 * c["score"] / mx)))
    return out


def find_highlights(cfg, segs, wav_path, duration):
    n, min_d, max_d = cfg["n"], cfg["min_d"], cfg["max_d"]
    if cfg["method"] == "Heuristic (बिना key, ऑफ़लाइन)":
        return heuristic_clips(wav_path, segs, n, min_d, max_d, duration)

    chunk_chars = 250000 if cfg["method"].startswith("Gemini") else 7000
    chunks = make_chunks(segs, chunk_chars)
    k = max(2, math.ceil(n * 2 / len(chunks)))
    raw = []
    for ch in chunks:
        prompt = build_prompt(ch, k, min_d, max_d, cfg["extra"])
        try:
            if cfg["method"].startswith("Gemini"):
                txt = call_gemini(cfg["gemini_key"], cfg["gemini_model"], prompt)
            else:
                txt = call_ollama(cfg["ollama_host"], cfg["ollama_model"], prompt)
        except Exception as e:
            st.warning(f"LLM कॉल में दिक्कत: {e}")
            continue
        for c in parse_json(txt).get("clips", []):
            try:
                s0, e0 = float(c["start"]), float(c["end"])
                if e0 <= s0 or s0 < 0 or s0 > duration:
                    continue
                s0, e0 = snap(s0, min(e0, duration), segs, min_d, max_d)
                if (e0 - s0) < min_d * 0.6:
                    continue
                raw.append({"start": s0, "end": e0, "title": str(c.get("title", "Clip"))[:90],
                            "hook": str(c.get("hook", "")), "reason": str(c.get("reason", "")),
                            "score": float(c.get("score", 50))})
            except Exception:
                continue
    if not raw:
        st.warning("LLM से क्लिप नहीं मिलीं, Heuristic इस्तेमाल कर रहा हूँ।")
        return heuristic_clips(wav_path, segs, n, min_d, max_d, duration)
    return pick(raw, n)


# ----------------------------------------------------------------------------
# 4) कैप्शन (ASS) + रेंडर (ffmpeg)
# ----------------------------------------------------------------------------

COLORS = {"Yellow": "00FFFF", "Green": "00FF00", "Cyan": "FFFF00",
          "Orange": "00A5FF", "Pink": "B469FF", "White": "FFFFFF"}


def ass_time(t):
    t = max(0.0, t)
    h = int(t // 3600)
    m = int(t % 3600 // 60)
    s = t % 60
    return f"{h}:{m:02d}:{s:05.2f}"


def build_ass(path, segs, c_start, c_end, W, H, o, title, speed):
    font, size = o["font"], o["cap_size"]
    align = {"Bottom": 2, "Middle": 5, "Top": 8}[o["cap_pos"]]
    marginv = int(H * 0.18) if o["cap_pos"] != "Middle" else 0
    hi = COLORS[o["cap_color"]]
    head = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {W}
PlayResY: {H}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Cap,{font},{size},&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,1,0,0,0,100,100,0,0,1,{max(3, size // 12)},2,{align},60,60,{marginv},1
Style: Title,{font},{int(size * 0.8)},&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,1,0,0,0,100,100,0,0,1,4,2,8,70,70,{int(H * 0.07)},1
Style: Mark,{font},{int(size * 0.45)},&H99FFFFFF,&H99FFFFFF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,2,0,9,40,40,{int(H * 0.03)},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    ev = []
    if o["cap_on"]:
        words = []
        for s in segs:
            if s["end"] < c_start or s["start"] > c_end:
                continue
            if s["words"]:
                for w in s["words"]:
                    if w["s"] >= c_start - 0.05 and w["e"] <= c_end + 0.3:
                        words.append(w)
            else:
                words.append({"w": s["text"], "s": max(s["start"], c_start), "e": min(s["end"], c_end)})
        wpl = max(1, o["cap_wpl"])
        for i in range(0, len(words), wpl):
            chunk = words[i:i + wpl]
            for j, w in enumerate(chunk):
                st_ = (w["s"] - c_start)
                en_ = (chunk[j + 1]["s"] - c_start) if j + 1 < len(chunk) else (w["e"] - c_start)
                if en_ <= st_:
                    en_ = st_ + 0.12
                parts = []
                for k2, w2 in enumerate(chunk):
                    t = w2["w"].replace("{", "(").replace("}", ")")
                    if o["cap_upper"]:
                        t = t.upper()
                    if k2 == j:
                        t = "{\\c&H" + hi + "&}" + t + "{\\c&HFFFFFF&}"
                    parts.append(t)
                ev.append(f"Dialogue: 0,{ass_time(st_ / speed)},{ass_time(en_ / speed)},Cap,,0,0,0,,{' '.join(parts)}")
    if o["title_on"] and title:
        ev.append(f"Dialogue: 1,{ass_time(0)},{ass_time(min(4.0, (c_end - c_start) / speed))},Title,,0,0,0,,"
                  "{\\fad(250,250)}" + title.replace("{", "(").replace("}", ")"))
    if o["mark"]:
        ev.append(f"Dialogue: 1,{ass_time(0)},{ass_time(24 * 3600 - 1)},Mark,,0,0,0,,{o['mark']}")
    Path(path).write_text(head + "\n".join(ev) + "\n", encoding="utf-8")


def render_clip(src, out_path, c, segs, o, workdir, idx, fast_preview=False):
    start, end = float(c["start"]), float(c["end"])
    dur = end - start
    W, H = o["W"], o["H"]
    speed = o["speed"]
    if fast_preview:
        W, H = int(W / 3), int(H / 3)

    # वीडियो फ़िल्टर
    if o["mode"] == "Blur background (fit)":
        fc = (f"[0:v]split=2[bg][fg];"
              f"[bg]scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},boxblur=25:5,eq=brightness=-0.08[b];"
              f"[fg]scale={W}:{H}:force_original_aspect_ratio=decrease[f];"
              f"[b][f]overlay=(W-w)/2:(H-h)/2,setsar=1[v0]")
    elif o["mode"] == "Center crop (fill)":
        fc = (f"[0:v]scale={W}:{H}:force_original_aspect_ratio=increase,"
              f"crop={W}:{H}:x='(iw-ow)*{o['xoff']}':y='(ih-oh)/2',setsar=1[v0]")
    else:  # Black bars
        fc = (f"[0:v]scale={W}:{H}:force_original_aspect_ratio=decrease,"
              f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:black,setsar=1[v0]")
    chain = []
    if o["zoom"] > 0:
        z = 1 + o["zoom"] / 100.0
        chain.append(f"scale=trunc(iw*{z}/2)*2:-2,crop={W}:{H}")
    if o["punchy"]:
        chain.append("eq=contrast=1.06:saturation=1.22")

    ass_name = None
    if (o["cap_on"] or o["title_on"] or o["mark"]) and not fast_preview:
        ass_name = f"cap_{idx}.ass"
        build_ass(workdir / ass_name, segs, start, end, W, H, o, c.get("title", ""), speed)
        chain.append(f"subtitles={ass_name}")
    if speed != 1.0:
        chain.append(f"setpts=PTS/{speed}")
    out_dur = dur / speed
    if o["fade"]:
        chain.append("fade=t=in:st=0:d=0.3")
        chain.append(f"fade=t=out:st={max(0, out_dur - 0.4):.2f}:d=0.4")
    fc += ";[v0]" + (",".join(chain) if chain else "null") + "[vout]"

    # ऑडियो फ़िल्टर
    a = []
    if speed != 1.0:
        a.append(f"atempo={speed}")
    if o["loud"]:
        a.append("loudnorm=I=-16:TP=-1.5:LRA=11")
    if o["fade"]:
        a.append("afade=t=in:st=0:d=0.3")
        a.append(f"afade=t=out:st={max(0, out_dur - 0.4):.2f}:d=0.4")
    if o["mute"]:
        a = ["volume=0"]
    a_chain = ",".join(a) if a else "anull"

    cmd = ["ffmpeg", "-y", "-ss", f"{start:.2f}", "-t", f"{dur:.2f}", "-i", str(src)]
    if o["music"] and not fast_preview:
        cmd += ["-stream_loop", "-1", "-i", str(o["music"])]
        fc += f";[0:a]{a_chain}[a0];[1:a]volume={o['music_vol']}[m];[a0][m]amix=inputs=2:duration=first:dropout_transition=0[aout]"
    else:
        fc += f";[0:a]{a_chain}[aout]"
    cmd += ["-filter_complex", fc, "-map", "[vout]", "-map", "[aout]",
            "-c:v", "libx264", "-preset", "ultrafast" if fast_preview else o["preset"],
            "-crf", "30" if fast_preview else str(o["crf"]), "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", str(out_path)]
    run(cmd, cwd=str(workdir))


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------

st.title("✂️ AI Shorts Clipper")
st.caption("लंबा वीडियो डालो → AI वायरल हाइलाइट ढूँढे → 9:16 शॉर्ट्स बनें। पूरा ओपन-सोर्स स्टैक।")

if not ffmpeg_ok():
    st.error("ffmpeg नहीं मिला। पहले ffmpeg इंस्टॉल करें (https://ffmpeg.org) और PATH में जोड़ें।")
    st.stop()

with st.sidebar:
    st.header("⚙️ सेटिंग्स")

    with st.expander("1) हाइलाइट खोजने का तरीका", expanded=True):
        method = st.selectbox("Method", [
            "Gemini (फ्री API key)",
            "Ollama (लोकल ओपन-सोर्स LLM)",
            "Heuristic (बिना key, ऑफ़लाइन)",
        ])
        gemini_key = gemini_model = ollama_host = ollama_model = ""
        if method.startswith("Gemini"):
            gemini_key = st.text_input("GEMINI_API_KEY", value=os.environ.get("GEMINI_API_KEY", ""), type="password",
                                       help="aistudio.google.com/apikey से फ्री मिलती है")
            gemini_model = st.text_input("Gemini model", value="gemini-2.5-flash")
        elif method.startswith("Ollama"):
            ollama_host = st.text_input("Ollama host", value="http://localhost:11434")
            ollama_model = st.text_input("Ollama model", value="qwen2.5:7b",
                                         help="पहले `ollama pull qwen2.5:7b` चलाएँ")
        n_clips = st.slider("कितनी क्लिप चाहिए", 3, 20, 10)
        min_d, max_d = st.slider("क्लिप की लंबाई (सेकंड)", 15, 90, (25, 60))
        extra = st.text_area("AI को ख़ास निर्देश (ऑप्शनल)", placeholder="जैसे: सिर्फ़ मज़ेदार और चौंकाने वाले मोमेंट चुनो")

    with st.expander("2) ट्रांसक्रिप्शन (Whisper, लोकल)"):
        wmodel = st.selectbox("Model size", ["tiny", "base", "small", "medium", "large-v3"], index=1,
                              help="बड़ा मॉडल = बेहतर हिंदी, पर धीमा। Streamlit Cloud पर 'base' या 'tiny' ही रखें (कम RAM)")
        lang = st.selectbox("भाषा", ["auto", "hi", "en", "ur", "bn", "ta", "te", "mr", "gu", "pa"])

    with st.expander("3) फ्रेम / आकार"):
        aspect = st.selectbox("Aspect", ["9:16 (Shorts/Reels)", "1:1 (Square)", "4:5 (Instagram)"])
        mode = st.selectbox("Reframe", ["Blur background (fit)", "Center crop (fill)", "Black bars"])
        xoff = st.slider("Crop की X-position (0=बाएँ, 1=दाएँ)", 0.0, 1.0, 0.5, 0.05) if mode.startswith("Center") else 0.5
        zoom = st.slider("Extra zoom %", 0, 30, 0)
        res = st.selectbox("Resolution (चौड़ाई)", [1080, 720], index=0)

    with st.expander("4) कैप्शन"):
        cap_on = st.checkbox("Auto captions (word-highlight)", True)
        font = st.text_input("Font", "Arial", help="हिंदी के लिए 'Noto Sans Devanagari' इंस्टॉल करके यहाँ लिखें")
        cap_size = st.slider("Caption size", 40, 130, 78)
        cap_wpl = st.slider("एक बार में शब्द", 1, 6, 3)
        cap_color = st.selectbox("Highlight color", list(COLORS.keys()))
        cap_pos = st.selectbox("Position", ["Bottom", "Middle", "Top"])
        cap_upper = st.checkbox("UPPERCASE (अंग्रेज़ी के लिए)", False)

    with st.expander("5) एडिटिंग"):
        title_on = st.checkbox("शुरू के 4 सेकंड में हेडलाइन दिखाओ", True)
        mark = st.text_input("Watermark / चैनल नाम", "")
        speed = st.slider("Speed", 0.8, 1.5, 1.0, 0.05)
        fade = st.checkbox("Fade in/out", True)
        punchy = st.checkbox("रंग थोड़े ज़्यादा चमकदार", True)
        loud = st.checkbox("आवाज़ बराबर करो (loudnorm)", True)
        mute = st.checkbox("ओरिजिनल आवाज़ बंद", False)
        music_file = st.file_uploader("बैकग्राउंड म्यूज़िक (ऑप्शनल)", type=["mp3", "wav", "m4a"])
        music_vol = st.slider("म्यूज़िक वॉल्यूम", 0.02, 0.6, 0.12, 0.02)
        preset = st.selectbox("Encode speed", ["veryfast", "fast", "medium"], index=0)
        crf = st.slider("Quality (कम=बेहतर)", 16, 28, 20)

    with st.expander("6) डाउनलोड"):
        max_h = st.selectbox("Max source quality", [1080, 720, 480, 1440, 2160], index=0)
        cookies = st.file_uploader("cookies.txt (ज़रूरत पड़े तो)", type=["txt"])

    if st.button("🧹 workspace साफ़ करो"):
        shutil.rmtree(WORK, ignore_errors=True)
        WORK.mkdir(exist_ok=True)
        st.session_state.pop("proj", None)
        st.success("साफ़ हो गया")

W_map = {"9:16 (Shorts/Reels)": (9, 16), "1:1 (Square)": (1, 1), "4:5 (Instagram)": (4, 5)}
aw, ah = W_map[aspect]
OUT_W = int(res)
OUT_H = int(round(OUT_W * ah / aw / 2) * 2)

st.info("⚠️ सिर्फ़ वही वीडियो इस्तेमाल करें जिन पर आपका हक़ है / जिनकी इजाज़त है / Creative Commons हैं। "
        "दूसरों के वीडियो की क्लिप अपने चैनल पर डालने से कॉपीराइट स्ट्राइक या चैनल बंद हो सकता है।")

tab_url, tab_file = st.tabs(["🔗 YouTube URL", "📁 अपनी वीडियो फाइल"])
with tab_url:
    url = st.text_input("YouTube वीडियो का URL", placeholder="https://www.youtube.com/watch?v=...")
with tab_file:
    up = st.file_uploader("वीडियो फाइल", type=["mp4", "mkv", "mov", "webm"])

if st.button("🔍 वीडियो Analyze करो", type="primary"):
    if method.startswith("Gemini") and not gemini_key:
        st.error("Gemini key डालें या Method बदलें।")
        st.stop()
    if not url and not up:
        st.error("URL डालें या फाइल अपलोड करें।")
        st.stop()
    pdir = WORK / f"p_{int(time.time())}"
    pdir.mkdir(parents=True, exist_ok=True)
    try:
        with st.status("काम चल रहा है...", expanded=True) as status:
            if up is not None and not url:
                st.write("📁 फाइल सेव हो रही है...")
                src = pdir / ("source" + Path(up.name).suffix)
                src.write_bytes(up.getbuffer())
                vtitle = Path(up.name).stem
            else:
                st.write("⬇️ वीडियो डाउनलोड हो रहा है...")
                bar = st.progress(0.0)
                cpath = None
                if cookies is not None:
                    cpath = str(pdir / "cookies.txt")
                    Path(cpath).write_bytes(cookies.getbuffer())
                src, vtitle = download_video(url, pdir, max_h, cpath, on_progress=lambda p: bar.progress(p))
            duration = get_duration(src)
            st.write(f"✅ वीडियो तैयार: {vtitle} ({fmt(duration)})")

            st.write("🎧 ऑडियो निकाल रहा हूँ...")
            wav = pdir / "audio.wav"
            extract_audio(src, wav)

            st.write("📝 बोले हुए शब्द लिख रहा हूँ (Whisper)... पहली बार मॉडल डाउनलोड होगा")
            bar2 = st.progress(0.0)
            segs, dlang = transcribe(wav, wmodel, lang, duration, on_progress=lambda p: bar2.progress(p))
            st.write(f"✅ भाषा: {dlang}, {len(segs)} सेगमेंट")

            st.write("🔥 वायरल हाइलाइट ढूँढ रहा हूँ...")
            cfg = dict(method=method, n=n_clips, min_d=min_d, max_d=max_d, extra=extra,
                       gemini_key=gemini_key, gemini_model=gemini_model,
                       ollama_host=ollama_host, ollama_model=ollama_model)
            clips = find_highlights(cfg, segs, wav, duration)
            clips = sorted(clips, key=lambda c: c["start"])
            status.update(label=f"तैयार! {len(clips)} क्लिप मिलीं", state="complete")
        st.session_state["proj"] = {"dir": str(pdir), "src": str(src), "title": vtitle,
                                    "segs": segs, "clips": clips, "duration": duration, "outs": []}
        for k in [k for k in st.session_state if k.startswith(("use_", "st_", "en_", "ti_"))]:
            del st.session_state[k]
    except Exception as e:
        st.error(f"गड़बड़: {e}")

proj = st.session_state.get("proj")
if proj:
    st.subheader(f"🎬 {proj['title']}  ·  {fmt(proj['duration'])}")
    st.caption("हर क्लिप का टाइम और टाइटल बदल सकते हैं। जो नहीं चाहिए उसका टिक हटा दें।")
    pdir = Path(proj["dir"])
    for i, c in enumerate(proj["clips"]):
        with st.expander(f"#{i + 1} · {fmt(c['start'])} → {fmt(c['end'])} · स्कोर {int(c['score'])} · {c['title']}",
                         expanded=(i < 3)):
            c1, c2 = st.columns([1, 3])
            with c1:
                st.checkbox("बनाओ", True, key=f"use_{i}")
                st.number_input("Start (सेकंड)", 0.0, float(proj["duration"]), float(c["start"]), 0.5, key=f"st_{i}")
                st.number_input("End (सेकंड)", 0.0, float(proj["duration"]), float(c["end"]), 0.5, key=f"en_{i}")
            with c2:
                st.text_input("टाइटल", c["title"], key=f"ti_{i}")
                if c.get("hook"):
                    st.write(f"🎣 **Hook:** {c['hook']}")
                if c.get("reason"):
                    st.write(f"💡 {c['reason']}")
                if st.button("👀 क्विक प्रीव्यू (छोटा/तेज़)", key=f"pv_{i}"):
                    try:
                        cc = {"start": st.session_state[f"st_{i}"], "end": st.session_state[f"en_{i}"],
                              "title": st.session_state[f"ti_{i}"]}
                        o_pv = dict(W=OUT_W, H=OUT_H, mode=mode, xoff=xoff, zoom=zoom, punchy=False, speed=1.0,
                                    fade=False, loud=False, mute=False, music=None, music_vol=0.1, preset="ultrafast",
                                    crf=30, cap_on=False, title_on=False, mark="", font=font, cap_size=cap_size,
                                    cap_wpl=cap_wpl, cap_color=cap_color, cap_pos=cap_pos, cap_upper=cap_upper)
                        pv = pdir / f"preview_{i}.mp4"
                        render_clip(proj["src"], pv, cc, proj["segs"], o_pv, pdir, i, fast_preview=True)
                        st.video(pv.read_bytes())
                    except Exception as e:
                        st.error(e)

    if st.button("🚀 चुनी हुई सारी क्लिप बनाओ", type="primary"):
        music_path = None
        if music_file is not None:
            music_path = pdir / ("music" + Path(music_file.name).suffix)
            music_path.write_bytes(music_file.getbuffer())
        o = dict(W=OUT_W, H=OUT_H, mode=mode, xoff=xoff, zoom=zoom, punchy=punchy, speed=speed, fade=fade,
                 loud=loud, mute=mute, music=music_path, music_vol=music_vol, preset=preset, crf=crf,
                 cap_on=cap_on, title_on=title_on, mark=mark.strip(), font=font, cap_size=cap_size,
                 cap_wpl=cap_wpl, cap_color=cap_color, cap_pos=cap_pos, cap_upper=cap_upper)
        outdir = pdir / "out"
        outdir.mkdir(exist_ok=True)
        todo = [i for i in range(len(proj["clips"])) if st.session_state.get(f"use_{i}", True)]
        prog = st.progress(0.0)
        outs, titles = [], []
        for n_, i in enumerate(todo):
            cc = {"start": st.session_state.get(f"st_{i}", proj["clips"][i]["start"]),
                  "end": st.session_state.get(f"en_{i}", proj["clips"][i]["end"]),
                  "title": st.session_state.get(f"ti_{i}", proj["clips"][i]["title"])}
            outp = outdir / f"short_{i + 1:02d}_{safe_name(cc['title'])}.mp4"
            try:
                render_clip(proj["src"], outp, cc, proj["segs"], o, outdir, i)
                outs.append(outp)
                titles.append(f"short_{i + 1:02d}: {cc['title']}")
            except Exception as e:
                st.error(f"क्लिप #{i + 1} फेल: {e}")
            prog.progress((n_ + 1) / max(1, len(todo)))
        if outs:
            (outdir / "titles.txt").write_text("\n".join(titles), encoding="utf-8")
            zp = pdir / "all_shorts.zip"
            with zipfile.ZipFile(zp, "w") as z:
                for f in outs:
                    z.write(f, f.name)
                z.write(outdir / "titles.txt", "titles.txt")
            proj["outs"] = [str(f) for f in outs]
            proj["zip"] = str(zp)
        st.success(f"{len(outs)} शॉर्ट्स तैयार!")

    if proj.get("outs"):
        st.subheader("✅ तैयार शॉर्ट्स")
        if proj.get("zip") and Path(proj["zip"]).exists():
            st.download_button("⬇️ सब एक ZIP में डाउनलोड करो", Path(proj["zip"]).read_bytes(),
                               file_name="all_shorts.zip", mime="application/zip")
        cols = st.columns(3)
        for k, f in enumerate(proj["outs"]):
            f = Path(f)
            if not f.exists():
                continue
            with cols[k % 3]:
                st.video(f.read_bytes())
                st.download_button(f"⬇️ {f.stem[:28]}", f.read_bytes(), file_name=f.name,
                                   mime="video/mp4", key=f"dl_{k}")
