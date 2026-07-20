"use client";

import { useRef, useState } from "react";
import { analyzePrescription, ApiError, type AnalyzePrescriptionResponse } from "@/lib/api";

interface Props {
  onResult: (result: AnalyzePrescriptionResponse) => void;
  onError: (message: string) => void;
  onStart: () => void;
}

export default function ReportUpload({ onResult, onError, onStart }: Props) {
  const [dragOver, setDragOver] = useState(false);
  const [loading, setLoading] = useState(false);
  const [fileName, setFileName] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  async function handleFile(file: File) {
    if (!file.name.toLowerCase().endsWith(".pdf")) {
      onError("Please upload a PDF file.");
      return;
    }
    setFileName(file.name);
    setLoading(true);
    onStart();
    try {
      const result = await analyzePrescription(file);
      onResult(result);
    } catch (err) {
      const message = err instanceof ApiError ? err.message : "Unexpected error analyzing the PDF.";
      onError(message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div
      className={`rounded-xl border-2 border-dashed p-10 text-center transition-colors ${
        dragOver ? "border-sky-400 bg-sky-950/30" : "border-slate-700 bg-slate-900/40"
      }`}
      onDragOver={(e) => {
        e.preventDefault();
        setDragOver(true);
      }}
      onDragLeave={() => setDragOver(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDragOver(false);
        const file = e.dataTransfer.files?.[0];
        if (file) void handleFile(file);
      }}
    >
      <input
        ref={inputRef}
        type="file"
        accept="application/pdf"
        className="hidden"
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) void handleFile(file);
        }}
      />
      <p className="text-lg font-medium text-slate-200">
        Drop a prescription PDF here, or
        <button
          type="button"
          className="ml-1 text-sky-400 underline underline-offset-2 hover:text-sky-300"
          onClick={() => inputRef.current?.click()}
          disabled={loading}
        >
          browse
        </button>
      </p>
      <p className="mt-1 text-sm text-slate-500">PDF only, up to 15MB. Text or scanned (OCR) documents supported.</p>
      {fileName && (
        <p className="mt-4 text-sm text-slate-400">
          {loading ? "Analyzing " : "Last file: "}
          <span className="text-slate-200">{fileName}</span>
          {loading && <span className="ml-2 animate-pulse">…</span>}
        </p>
      )}
    </div>
  );
}
