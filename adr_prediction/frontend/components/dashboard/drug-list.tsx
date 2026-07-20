import type { AnalyzePrescriptionResponse } from "@/lib/api";

export default function DrugList({ result }: { result: AnalyzePrescriptionResponse }) {
  if (result.drug_list.length === 0) {
    return (
      <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-6">
        <p className="text-sm font-medium text-slate-300">Detected Drugs</p>
        <p className="mt-2 text-sm text-slate-500">No recognized drug names were found in this document.</p>
      </div>
    );
  }

  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-6">
      <p className="text-sm font-medium text-slate-300">Detected Drugs ({result.drug_list.length})</p>
      <div className="mt-3 overflow-x-auto">
        <table className="w-full text-left text-sm">
          <thead>
            <tr className="text-slate-500">
              <th className="pb-2 pr-4 font-normal">Drug</th>
              <th className="pb-2 pr-4 font-normal">Dosage</th>
              <th className="pb-2 pr-4 font-normal">Frequency</th>
              <th className="pb-2 font-normal">Risk</th>
            </tr>
          </thead>
          <tbody>
            {result.drug_list.map((drug) => (
              <tr key={drug.name} className="border-t border-slate-800">
                <td className="py-2 pr-4 text-slate-100">{drug.name}</td>
                <td className="py-2 pr-4 text-slate-400">
                  {drug.dosage_mg !== null ? `${drug.dosage_mg} mg` : "—"}
                </td>
                <td className="py-2 pr-4 text-slate-400">{drug.frequency ?? "—"}</td>
                <td className="py-2">
                  {drug.high_risk ? (
                    <span className="rounded-full bg-red-500/15 px-2 py-0.5 text-xs text-red-400">
                      High-risk / narrow index
                    </span>
                  ) : (
                    <span className="text-xs text-slate-500">standard</span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
