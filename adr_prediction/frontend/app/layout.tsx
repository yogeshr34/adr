import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "ADR Prediction Dashboard",
  description: "Upload a prescription PDF and get an adverse drug reaction risk report.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen antialiased">{children}</body>
    </html>
  );
}
