import whisper

# tải model (small/medium/large)
model = whisper.load_model("small")

result = model.transcribe("test.mp4", language="vi")

print("📜 Văn bản nhận được:")
print(result["text"])
