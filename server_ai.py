import os
import struct
from io import BytesIO

from flask import Flask, request, Response, jsonify

from google import genai
from google.genai import types

# ========= AYARLAR =========

# Gemini Developer API key (ai.google.dev'den al)
GEMINI_API_KEY = "AIzaSyBUQ0AWfn1SKRj8nwzyY_NA109Key2-CI4"

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY ortam değişkeni set değil, önce onu ayarla.")

# Kullanacağın modeller
TEXT_MODEL = "gemini-1.5-flash"       # Konuşmayı anlamak ve yanıt yazmak için
TTS_MODEL  = "gemini-1.5-flash-tts"   # Metni sese çevirmek için  :contentReference[oaicite:1]{index=1}

# ESP32'dEN GELEN SESİN FORMATINI BUNA GÖRE KABUL EDİYORUZ:
#  - 16-bit signed PCM
#  - mono
#  - 16000 Hz
ESP_INPUT_SAMPLE_RATE = 16000

# TTS ÇIKIŞI (Gemini tarafı) -> audio/L16;rate=24000
GEMINI_TTS_SAMPLE_RATE = 24000

# ===========================

app = Flask(__name__)
client = genai.Client(api_key=GEMINI_API_KEY)


# -------- Yardımcı: PCM -> WAV (mono, 16-bit, 16kHz) --------
def pcm16le_to_wav(pcm_bytes: bytes, sample_rate: int = ESP_INPUT_SAMPLE_RATE) -> bytes:
    """
    ESP32'den gelen raw PCM verisini (16-bit mono) basit bir WAV dosyasına çevirir.
    """
    num_channels = 1
    bits_per_sample = 16
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8
    data_size = len(pcm_bytes)
    riff_chunk_size = 36 + data_size

    buf = BytesIO()

    # RIFF header
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", riff_chunk_size))
    buf.write(b"WAVE")

    # fmt chunk
    buf.write(b"fmt ")
    buf.write(struct.pack("<I", 16))               # Subchunk1Size
    buf.write(struct.pack("<H", 1))                # AudioFormat = PCM
    buf.write(struct.pack("<H", num_channels))     # NumChannels
    buf.write(struct.pack("<I", sample_rate))      # SampleRate
    buf.write(struct.pack("<I", byte_rate))        # ByteRate
    buf.write(struct.pack("<H", block_align))      # BlockAlign
    buf.write(struct.pack("<H", bits_per_sample))  # BitsPerSample

    # data chunk
    buf.write(b"data")
    buf.write(struct.pack("<I", data_size))
    buf.write(pcm_bytes)

    return buf.getvalue()


# -------- Gemini: Audio -> Text --------
def transcribe_and_answer(wav_bytes: bytes) -> str:
    """
    Gelen konuşmayı Gemini ile anlayıp Türkçe, kısa ve samimi bir cevap döndürür.
    """
    audio_part = types.Part.from_bytes(
        data=wav_bytes,
        mime_type="audio/wav",   # Gemini audio/wav kabul ediyor :contentReference[oaicite:2]{index=2}
    )

    system_prompt = (
        "Sen genç bir mühendislik öğrencisine yardım eden, "
        "samimi ama küfür etmeyen Türkçe bir sesli asistansın. "
        "Kısa, net ve doğal cevap ver. Gereksiz uzatma."
    )

    resp = client.models.generate_content(
        model=TEXT_MODEL,
        contents=[
            types.Part.from_text(system_prompt),
            audio_part,
        ],
    )

    # python-genai genelde .text alanına tekleştiriyor
    answer = (resp.text or "").strip()
    if not answer:
        # Yedek: candidate içinden çek
        if resp.candidates:
            parts = resp.candidates[0].content.parts
            text_chunks = [p.text for p in parts if hasattr(p, "text") and p.text]
            answer = " ".join(text_chunks).strip()

    return answer or "Seni duyamadım, tekrarlar mısın?"


# -------- Gemini: Text -> Audio (TTS) --------
def tts_from_text(text: str) -> bytes:
    """
    Gemini TTS ile metni 16-bit PCM (audio/L16;rate=24000) olarak üretir.
    Dönüş: raw PCM (WAV değil), direkt ESP32'nin çalabileceği format.
    """
    # TTS için prompt: stil vs.
    tts_prompt = (
        "Sen samimi, doğal bir Türkçe ses tonuyla konuşan tek konuşmacılı bir asistansın. "
        "Arkadaşına konuşur gibi oku."
    )

    config = types.GenerateContentConfig(
        response_mime_type="audio/L16;rate=24000"  # PCM 16-bit, 24kHz :contentReference[oaicite:3]{index=3}
    )

    resp = client.models.generate_content(
        model=TTS_MODEL,
        contents=[
            types.Part.from_text(tts_prompt),
            types.Part.from_text(text),
        ],
        config=config,
    )

    # Audio response inline_data olarak gelir
    audio_bytes = b""
    if resp.candidates:
        for part in resp.candidates[0].content.parts:
            if getattr(part, "inline_data", None) and part.inline_data.data:
                audio_bytes += part.inline_data.data

    if not audio_bytes:
        raise RuntimeError("Gemini TTS boş ses döndürdü.")

    return audio_bytes


# ================== HTTP ENDPOINTLER ==================

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/audio", methods=["POST"])
def handle_audio():
    """
    ESP32 buraya:
      - Gövde: raw PCM16 mono 16kHz (Content-Type: application/octet-stream)
    gönderiyor.

    Backend:
      - PCM -> WAV
      - Gemini ile anla & cevap üret (Türkçe metin)
      - Gemini TTS ile cevap metnini sese çevir (audio/L16;rate=24000)
      - Aynen raw PCM olarak geri yolla.
    """
    pcm_bytes = request.data
    if not pcm_bytes:
        return "No audio data", 400

    try:
        # 1) PCM -> WAV
        wav_bytes = pcm16le_to_wav(pcm_bytes, sample_rate=ESP_INPUT_SAMPLE_RATE)

        # 2) Audio understanding: konuşmayı anla + cevap üret
        answer_text = transcribe_and_answer(wav_bytes)
        print(f"[Gemini TEXT] {answer_text}")

        # 3) Text-to-Speech: cevabı ses yap
        tts_pcm = tts_from_text(answer_text)

        # 4) ESP32'ye raw PCM olarak geri yolla
        # ESP tarafında hoparlör I2S sample_rate = 24000 yapman gerekiyor!
        return Response(
            tts_pcm,
            mimetype="audio/L16;rate=24000"
        )

    except Exception as e:
        print("HATA:", e)
        return f"Server error: {e}", 500


if __name__ == "__main__":
    # Örn: python server_ai.py
    # Varsayılan: 0.0.0.0:5000
    app.run(host="0.0.0.0", port=5000, debug=True)

