// Fold the Vite build (dist/index.html + hashed JS/CSS assets) into a single
// self-contained HTML file that can be embedded into the standalone binary.
import { readFileSync, writeFileSync, existsSync } from "node:fs";
import path from "node:path";

const distDir = path.resolve(import.meta.dir, "..", "dist");
const indexPath = path.join(distDir, "index.html");

if (!existsSync(indexPath)) {
  console.error("dist/index.html not found. Run `bun run build` first.");
  process.exit(1);
}

function assetContents(url: string): string {
  const file = path.join(distDir, url.replace(/^\//, ""));
  return readFileSync(file, "utf8");
}

let html = readFileSync(indexPath, "utf8");

// Drop preloads — the referenced chunks are inlined below and won't exist.
html = html.replace(/<link[^>]*rel="modulepreload"[^>]*>\s*/g, "");

// Inline stylesheets.
html = html.replace(/<link[^>]*rel="stylesheet"[^>]*href="([^"]+)"[^>]*>/g, (_match, href: string) => {
  return `<style>\n${assetContents(href)}\n</style>`;
});

// Inline module scripts. Escaping any literal </script> inside the bundle keeps
// the surrounding <script> tag from closing early.
html = html.replace(/<script([^>]*)\ssrc="([^"]+)"([^>]*)><\/script>/g, (_match, _pre: string, src: string) => {
  const code = assetContents(src).replace(/<\/script>/g, "<\\/script>");
  return `<script type="module">\n${code}\n</script>`;
});

const remainingAssetRef = /(?:src|href)="\/assets\//.test(html);
if (remainingAssetRef) {
  console.error("Inlining incomplete: an /assets/ reference remains in the HTML.");
  process.exit(1);
}

// Emit with a .txt extension so it embeds via the text loader rather than
// Bun's default HTML-bundle loader (which would try to re-process assets).
const outPath = path.join(distDir, "standalone.html.txt");
writeFileSync(outPath, html);
console.log(`Wrote ${outPath} (${(html.length / 1024).toFixed(1)} KiB, self-contained).`);
