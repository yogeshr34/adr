import type { AnalyzePrescriptionResponse } from "@/lib/api";

export default function InteractionsPanel({ result }: { result: AnalyzePrescriptionResponse }) {
  if (result.interactions.length === 0) {
    return (
      <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-6">
        <p className="text-sm font-medium text-slate-300">Drug-Drug Interactions</p>
        <p className="mt-2 text-sm text-emerald-400">No known dangerous interactions detected among identified drugs.</p>
      </div>
    );
  }

  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-6">
      <p className="text-sm font-medium text-slate-300">Drug-Drug Interactions ({result.interactions.length})</p>
      <ul className="mt-3 space-y-3">
        {result.interactions.map((interaction) => (
          <li key={interaction.drugs.join("+")} className="rounded-lg border border-red-500/20 bg-red-950/20 p-3">
            <div className="flex items-center justify-between">
              <span className="font-medium text-red-300">{interaction.drugs.join(" + ")}</span>
              <span className="text-xs text-red-400">severity {interaction.severity.toFixed(2)}</span>
            </div>
            <p className="mt-1 text-sm text-slate-400">{interaction.reason}</p>
          </li>
        ))}
      </ul>
    </div>
  );
}
