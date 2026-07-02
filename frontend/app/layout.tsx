import type { Metadata } from "next";
import { Space_Grotesk, JetBrains_Mono } from "next/font/google";
import "./globals.css";
import { Nav } from "@/components/nav";

const space = Space_Grotesk({
  subsets: ["latin"],
  variable: "--font-space",
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
    <html lang="en" className={`dark ${space.variable} ${mono.variable}`}>
      <body className="grain font-sans antialiased" suppressHydrationWarning>
        <Nav />
        <main className="relative z-[1] mx-auto max-w-[1440px] px-5 pb-24 pt-6 sm:px-8">
          {children}
        </main>
      </body>
    </html>
  );
}
