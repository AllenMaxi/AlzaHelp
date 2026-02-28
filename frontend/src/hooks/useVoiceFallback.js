import { useRef, useCallback } from "react";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;

/**
 * Fallback voice input using MediaRecorder + Whisper API
 * for browsers where Web Speech API is unreliable (iOS Safari).
 */
export function useVoiceFallback() {
  const mediaRecorderRef = useRef(null);
  const chunksRef = useRef([]);

  const isNativeAvailable = useCallback(() => {
    const ua = navigator.userAgent;
    const isIOS =
      /iPad|iPhone|iPod/.test(ua) ||
      (navigator.platform === "MacIntel" && navigator.maxTouchPoints > 1);
    if (isIOS) return false;
    return "webkitSpeechRecognition" in window || "SpeechRecognition" in window;
  }, []);

  const startRecording = useCallback(async () => {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const mimeType = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
      ? "audio/webm;codecs=opus"
      : "audio/webm";
    const recorder = new MediaRecorder(stream, { mimeType });
    chunksRef.current = [];
    recorder.ondataavailable = (e) => {
      if (e.data.size > 0) chunksRef.current.push(e.data);
    };
    mediaRecorderRef.current = recorder;
    recorder.start(1000);
  }, []);

  const stopAndTranscribe = useCallback(async () => {
    return new Promise((resolve, reject) => {
      const recorder = mediaRecorderRef.current;
      if (!recorder || recorder.state === "inactive") {
        resolve(null);
        return;
      }

      recorder.onstop = async () => {
        const blob = new Blob(chunksRef.current, { type: "audio/webm" });
        // Stop all tracks to release mic
        recorder.stream.getTracks().forEach((t) => t.stop());

        const formData = new FormData();
        formData.append("file", blob, "recording.webm");

        try {
          const res = await fetch(`${BACKEND_URL}/api/voice/transcribe`, {
            method: "POST",
            credentials: "include",
            body: formData,
          });
          if (res.ok) {
            const data = await res.json();
            resolve(data.text);
          } else {
            resolve(null);
          }
        } catch (e) {
          reject(e);
        }
      };
      recorder.stop();
    });
  }, []);

  const isRecording = useCallback(() => {
    return (
      mediaRecorderRef.current &&
      mediaRecorderRef.current.state === "recording"
    );
  }, []);

  return { isNativeAvailable, startRecording, stopAndTranscribe, isRecording };
}
