// Electron 外壳：自动拉起 Python 服务，再把窗口指过去。
const { app, BrowserWindow, shell } = require("electron");
const { spawn } = require("child_process");
const path = require("path");
const http = require("http");

const HOST = "127.0.0.1";
const PORT = process.env.AILAB_PORT ? Number(process.env.AILAB_PORT) : 8123;
const PROJECT_ROOT = path.resolve(__dirname, "..", "..");

let serverProcess = null;
let mainWindow = null;

function pythonCommand() {
  return process.platform === "win32" ? "python" : "python3";
}

function startBackend() {
  if (process.env.AILAB_EXTERNAL_SERVER === "1") {
    console.log("[electron] 使用外部服务，跳过内置启动");
    return;
  }
  const args = ["run_server.py", "--host", HOST, "--port", String(PORT), "--no-fallback-port"];
  serverProcess = spawn(pythonCommand(), args, {
    cwd: PROJECT_ROOT,
    env: { ...process.env, PYTHONUTF8: "1" },
    stdio: ["ignore", "pipe", "pipe"],
  });
  serverProcess.stdout.on("data", (data) => process.stdout.write(`[server] ${data}`));
  serverProcess.stderr.on("data", (data) => process.stderr.write(`[server] ${data}`));
  serverProcess.on("exit", (code) => console.log(`[electron] 服务退出，code=${code}`));
}

function waitForServer(url, timeoutMs = 20000) {
  const deadline = Date.now() + timeoutMs;
  return new Promise((resolve, reject) => {
    const attempt = () => {
      const request = http.get(url, (response) => {
        response.resume();
        resolve();
      });
      request.on("error", () => {
        if (Date.now() > deadline) {
          reject(new Error("等待后端服务超时"));
        } else {
          setTimeout(attempt, 400);
        }
      });
    };
    attempt();
  });
}

async function createWindow() {
  const url = `http://${HOST}:${PORT}/`;
  try {
    await waitForServer(`${url}api/health`);
  } catch (error) {
    console.error(`[electron] ${error.message}`);
  }

  mainWindow = new BrowserWindow({
    width: 1320,
    height: 880,
    minWidth: 720,
    minHeight: 560,
    backgroundColor: "#f4f5f7",
    title: "AI Lab",
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });

  mainWindow.loadURL(url);
  mainWindow.webContents.setWindowOpenHandler(({ url: target }) => {
    shell.openExternal(target);
    return { action: "deny" };
  });
}

app.whenReady().then(() => {
  startBackend();
  createWindow();
  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") app.quit();
});

app.on("before-quit", () => {
  if (serverProcess) {
    console.log("[electron] 关闭后端服务");
    serverProcess.kill();
  }
});
