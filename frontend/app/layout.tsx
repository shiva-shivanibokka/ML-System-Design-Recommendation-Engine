import type { Metadata } from "next";
import { Bricolage_Grotesque, Manrope, JetBrains_Mono } from "next/font/google";
import "./globals.css";
import { Nav } from "@/components/nav";

const bricolage = Bricolage_Grotesque({
  subsets: ["latin"],
  variable: "--font-bricolage",
  display: "swap",
});
const manrope = Manrope({
  subsets: ["latin"],
  variable: "--font-manrope",
  display: "swap",
});
const mono = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-mono",
  display: "swap",
});

export const metadata: Metadata = {
  title: "RecSys — Real-Time Recommendation Engine",
  description:
    "A production-shaped movie recommender: FAISS two-stage retrieval, NeuMF + SVD ranking, Thompson-Sampling bandit, drift monitoring. MovieLens 1M.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`dark ${bricolage.variable} ${manrope.variable} ${mono.variable}`}>
      <body className="font-sans antialiased" suppressHydrationWarning>
        <div className="aurora" aria-hidden>
          <span className="b1" />
          <span className="b2" />
          <span className="b3" />
          <span className="b4" />
        </div>
        <Nav />
        <main className="relative z-[1] mx-auto max-w-[1480px] px-5 pb-24 pt-8 sm:px-8">
          {children}
        </main>
      </body>
    </html>
  );
}
