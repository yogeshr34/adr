"use client";

import { useState } from "react";
import ReportUpload from "@/components/dashboard/report-upload";
import RiskSummary from "@/components/dashboard/risk-summary";
import DrugList from "@/components/dashboard/drug-list";
import InteractionsPanel from "@/components/dashboard/interactions-panel";
import ShapChart from "@/components/dashboard/shap-chart";
import type { AnalyzePrescriptionResponse } from "@/lib/api";

export default function Home() {
  const [result, setResult] = useState<AnalyzePrescriptionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  return (
    <main className="mx-auto max-w-4xl px-6 py-12">
      <header className="mb-8">
        <h1 className="text-2xl font-bold text-slate-50">ADR Prediction Dashboard</h1>
        <p className="mt-1 text-slate-400">
          Upload a prescription PDF. Drugs, dosages, and patient context are extracted
          automatically and run through the ADR risk model — nothing below is mock data.
        </p>
      </header>

      <ReportUpload
        onStart={() => {
          setLoading(true);
          setError(null);
        }}
        onResult={(r) => {
          setResult(r);
          setError(null);
          setLoading(false);
        }}
        onError={(message) => {
          setError(message);
          setResult(null);
          setLoading(false);
        }}
      />

      {error && (
        <div className="mt-6 rounded-lg border border-red-500/30 bg-red-950/30 p-4 text-red-300">
          {error}
        </div>
      )}

      {loading && !result && !error && (
        <div className="mt-6 text-center text-slate-500">Analyzing prescription…</div>
      )}

      {result && (
        <div className="mt-8 space-y-6">
          <RiskSummary result={result} />
          <DrugList result={result} />
          <InteractionsPanel result={result} />
          <ShapChart result={result} />
          <p className="text-center text-xs text-slate-600">
            Extraction method: {result.extraction_method} · This tool is decision support only
            and does not replace clinical judgment.
          </p>
        </div>
      )}
    </main>
  );
}
