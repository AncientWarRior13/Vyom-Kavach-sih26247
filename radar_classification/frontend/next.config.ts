import type { NextConfig } from "next";
import path from "path";

const nextConfig: NextConfig = {
  turbopack: {
    // app/globals.css imports ../../../packages/design-tokens/tokens.css
    // (shared tactical theme vars) — Turbopack otherwise refuses to read
    // outside this directory's own configured root.
    root: path.resolve(__dirname, "../.."),
  },
};

export default nextConfig;
