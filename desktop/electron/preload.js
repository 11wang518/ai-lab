// 只暴露一个只读的版本信息，其余能力保持关闭。
const { contextBridge } = require("electron");

contextBridge.exposeInMainWorld("aiLab", {
  desktop: true,
  platform: process.platform,
});
