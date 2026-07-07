// Opens a native folder/file picker on the host and returns the chosen absolute
// path. The frontend runs in a browser, which for security never exposes real
// filesystem paths from <input type=file> — so the picker must run server-side
// and shell out to each OS's dialog tool. Pure command builders + the output
// parser are exported for unit tests; pickPath() wires them to a real spawn.
import { spawn } from "node:child_process";

export type PickMode = "folder" | "file";

export interface PickOptions {
  mode: PickMode;
  title?: string;
  gguf?: boolean; // narrow file picking to .gguf models
}

export interface DialogCommand {
  cmd: string;
  args: string[];
}

export interface RunResult {
  code: number | null;
  stdout: string;
}

export type DialogRunner = (command: DialogCommand) => Promise<RunResult>;

const GGUF_LABEL = "GGUF models";

function psEscape(value: string): string {
  return value.replaceAll("'", "''");
}

// ---- Windows (PowerShell + WinForms) ----

export function windowsCommand(opts: PickOptions): DialogCommand {
  const title = psEscape(opts.title ?? (opts.mode === "folder" ? "Select folder" : "Select file"));
  const script =
    opts.mode === "folder"
      ? [
          "Add-Type -AssemblyName System.Windows.Forms | Out-Null",
          "$owner = New-Object System.Windows.Forms.Form; $owner.TopMost = $true",
          "$dlg = New-Object System.Windows.Forms.FolderBrowserDialog",
          `$dlg.Description = '${title}'`,
          "if ($dlg.ShowDialog($owner) -eq [System.Windows.Forms.DialogResult]::OK) { [Console]::Out.Write($dlg.SelectedPath) }"
        ].join("; ")
      : [
          "Add-Type -AssemblyName System.Windows.Forms | Out-Null",
          "$owner = New-Object System.Windows.Forms.Form; $owner.TopMost = $true",
          "$dlg = New-Object System.Windows.Forms.OpenFileDialog",
          `$dlg.Title = '${title}'`,
          `$dlg.Filter = '${opts.gguf ? "GGUF models (*.gguf)|*.gguf|All files (*.*)|*.*" : "All files (*.*)|*.*"}'`,
          "if ($dlg.ShowDialog($owner) -eq [System.Windows.Forms.DialogResult]::OK) { [Console]::Out.Write($dlg.FileName) }"
        ].join("; ");
  return { cmd: "powershell", args: ["-NoProfile", "-STA", "-Command", script] };
}

// ---- macOS (AppleScript) ----

export function macCommand(opts: PickOptions): DialogCommand {
  const prompt = (opts.title ?? (opts.mode === "folder" ? "Select folder" : "Select file")).replaceAll('"', '\\"');
  const chooser = opts.mode === "folder" ? "choose folder" : "choose file";
  const script = `POSIX path of (${chooser} with prompt "${prompt}")`;
  return { cmd: "osascript", args: ["-e", script] };
}

// ---- Linux (zenity / kdialog) ----

export function zenityCommand(opts: PickOptions): DialogCommand {
  const args = ["--file-selection", "--title", opts.title ?? "Select"];
  if (opts.mode === "folder") {
    args.push("--directory");
  } else if (opts.gguf) {
    args.push("--file-filter", `${GGUF_LABEL} | *.gguf`);
  }
  return { cmd: "zenity", args };
}

export function kdialogCommand(opts: PickOptions): DialogCommand {
  if (opts.mode === "folder") {
    return { cmd: "kdialog", args: ["--getexistingdirectory", ".", "--title", opts.title ?? "Select"] };
  }
  const args = ["--getopenfilename", ".", opts.gguf ? "*.gguf" : "*"];
  return { cmd: "kdialog", args: ["--title", opts.title ?? "Select", ...args] };
}

// Trim whitespace/newlines and drop a single trailing slash (macOS folder picks
// return one). Empty output means the user cancelled.
export function parseDialogOutput(stdout: string): string | null {
  const trimmed = stdout.trim();
  if (!trimmed) {
    return null;
  }
  return trimmed.length > 1 ? trimmed.replace(/[/\\]$/u, "") : trimmed;
}

const defaultRunner: DialogRunner = (command) =>
  new Promise((resolve, reject) => {
    // windowsHide hides the transient console; the dialog window still shows.
    const child = spawn(command.cmd, command.args, { windowsHide: true });
    let stdout = "";
    child.stdout.on("data", (chunk: Buffer) => {
      stdout += chunk.toString("utf8");
    });
    child.once("error", reject);
    child.once("close", (code) => resolve({ code, stdout }));
  });

export interface PickDeps {
  platform?: NodeJS.Platform;
  run?: DialogRunner;
}

const MISSING_LINUX_TOOL =
  "No native file dialog found. Install zenity (or kdialog), or type the path manually.";

export async function pickPath(opts: PickOptions, deps: PickDeps = {}): Promise<string | null> {
  const platform = deps.platform ?? process.platform;
  const run = deps.run ?? defaultRunner;

  if (platform === "win32") {
    return parseDialogOutput((await run(windowsCommand(opts))).stdout);
  }
  if (platform === "darwin") {
    // Cancel exits non-zero with empty stdout → parsed as null.
    return parseDialogOutput((await run(macCommand(opts))).stdout);
  }

  // Linux: prefer zenity, fall back to kdialog; a missing binary rejects with ENOENT.
  for (const command of [zenityCommand(opts), kdialogCommand(opts)]) {
    try {
      return parseDialogOutput((await run(command)).stdout);
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === "ENOENT") {
        continue;
      }
      throw error;
    }
  }
  throw new Error(MISSING_LINUX_TOOL);
}
