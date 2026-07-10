// Preview harness entry: run the app on a fixed side port so verification
// never collides with a user-started instance on the default 4187.
process.env.LOCALLLAMA_PORT = "4218";
await import("../server/index.ts");
