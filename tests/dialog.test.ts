import { describe, expect, test } from "bun:test";

import {
  kdialogCommand,
  macCommand,
  parseDialogOutput,
  pickPath,
  windowsCommand,
  zenityCommand,
  type DialogCommand,
  type RunResult
} from "../server/dialog";

describe("windowsCommand", () => {
  test("folder picker uses FolderBrowserDialog with the title", () => {
    const { cmd, args } = windowsCommand({ mode: "folder", title: "Pick a folder" });
    expect(cmd).toBe("powershell");
    expect(args).toContain("-STA");
    expect(args).toContain("-Command");
    const script = args[args.length - 1];
    expect(script).toContain("FolderBrowserDialog");
    expect(script).toContain("Pick a folder");
  });

  test("file picker uses OpenFileDialog and a .gguf filter when requested", () => {
    const script = windowsCommand({ mode: "file", gguf: true }).args.at(-1)!;
    expect(script).toContain("OpenFileDialog");
    expect(script).toContain("*.gguf");
  });

  test("single quotes in the title are escaped", () => {
    const script = windowsCommand({ mode: "folder", title: "Ann's models" }).args.at(-1)!;
    expect(script).toContain("Ann''s models");
  });
});

describe("macCommand", () => {
  test("maps mode to the AppleScript chooser", () => {
    expect(macCommand({ mode: "folder" }).args.at(-1)).toContain("choose folder");
    expect(macCommand({ mode: "file" }).args.at(-1)).toContain("choose file");
    expect(macCommand({ mode: "folder" }).cmd).toBe("osascript");
  });
});

describe("linux commands", () => {
  test("zenity folder adds --directory; file adds a gguf filter", () => {
    expect(zenityCommand({ mode: "folder" }).args).toContain("--directory");
    const fileArgs = zenityCommand({ mode: "file", gguf: true }).args;
    expect(fileArgs).toContain("--file-filter");
    expect(fileArgs.join(" ")).toContain("*.gguf");
  });

  test("kdialog folder uses getexistingdirectory", () => {
    expect(kdialogCommand({ mode: "folder" }).args).toContain("--getexistingdirectory");
    expect(kdialogCommand({ mode: "file", gguf: true }).args).toContain("*.gguf");
  });
});

describe("parseDialogOutput", () => {
  test("trims whitespace and a trailing slash", () => {
    expect(parseDialogOutput("C:\\models\\llama\r\n")).toBe("C:\\models\\llama");
    expect(parseDialogOutput("/Users/me/llama/")).toBe("/Users/me/llama");
    expect(parseDialogOutput("/home/me/model.gguf\n")).toBe("/home/me/model.gguf");
  });

  test("empty output (cancel) is null", () => {
    expect(parseDialogOutput("")).toBeNull();
    expect(parseDialogOutput("   \n")).toBeNull();
  });
});

describe("pickPath", () => {
  const run = (result: RunResult) => async () => result;

  test("returns the chosen path on Windows", async () => {
    const picked = await pickPath(
      { mode: "folder" },
      { platform: "win32", run: run({ code: 0, stdout: "C:\\llama.cpp" }) }
    );
    expect(picked).toBe("C:\\llama.cpp");
  });

  test("cancel on macOS yields null", async () => {
    const picked = await pickPath({ mode: "file" }, { platform: "darwin", run: run({ code: 1, stdout: "" }) });
    expect(picked).toBeNull();
  });

  test("Linux falls back from zenity to kdialog when zenity is missing", async () => {
    const calls: string[] = [];
    const runner = async (command: DialogCommand): Promise<RunResult> => {
      calls.push(command.cmd);
      if (command.cmd === "zenity") {
        throw Object.assign(new Error("not found"), { code: "ENOENT" });
      }
      return { code: 0, stdout: "/home/me/llama.cpp" };
    };
    const picked = await pickPath({ mode: "folder" }, { platform: "linux", run: runner });
    expect(picked).toBe("/home/me/llama.cpp");
    expect(calls).toEqual(["zenity", "kdialog"]);
  });

  test("Linux with no dialog tool throws a helpful error", async () => {
    const runner = async (): Promise<RunResult> => {
      throw Object.assign(new Error("not found"), { code: "ENOENT" });
    };
    await expect(pickPath({ mode: "folder" }, { platform: "linux", run: runner })).rejects.toThrow(/zenity/);
  });
});
