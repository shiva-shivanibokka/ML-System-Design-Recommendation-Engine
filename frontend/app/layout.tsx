import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import { Nav } from "@/components/nav";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "RecSys — Real-Time Recommendation Engine",
  description:
    "Netflix-style recommendation system: FAISS two-stage retrieval, NeuMF, SVD, Thompson Sampling A/B, Feast feature store.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="dark">
      <body className={inter.className}>
        <div className="flex min-h-screen">
          <Nav />
          <main className="ml-60 flex-1 overflow-y-auto">{children}</main>
        </div>
      </body>
    </html>
  );
}
