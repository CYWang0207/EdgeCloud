# EdgeCloud Demo Web Prototype

离线静态演示雏形。直接打开 `index.html` 即可使用，无需安装依赖或启动服务。

## 线上访问

- 演示网站：<https://cywang0207.github.io/EdgeCloud-Demo/>
- 公共网页仓库：<https://github.com/CYWang0207/EdgeCloud-Demo>

项目代码继续保存在私有 `EdgeCloud` 仓库中，公共仓库只发布 `demo-web` 目录下的网站文件。

## 发布更新

在 `EdgeCloud` 仓库提交网页修改后，从仓库根目录执行：

```bash
git subtree push --prefix=demo-web https://github.com/CYWang0207/EdgeCloud-Demo.git main
```

GitHub Pages 会在推送后自动重新构建，通常一分钟内更新线上版本。

## 已实现

- 电影化项目首页
- 渐变黑色导航与侧边分区菜单
- 首页、菜单和团队区 GitHub 公共仓库入口
- BoxCars 与 ModelNet40 双场景四视图演示台
- 为展示效果扩展至 15 秒的端-边-云回放状态机
- 光照、运动模糊、传感器噪声切换
- 播放、暂停、重置、拖动或点击阶段时间轴和 JSON 导出
- 动态视角选择、决策因果链和分级事件日志
- Adapter 知识下发流程
- 准确率、连续性、冲突仲裁和时延证据区
- ModelNet40 `airplane_0627` 官方测试四视图与交互漂移效果
- 东南大学校徽与团队信息区
- 桌面和移动端响应式布局

## 替换真实素材

1. BoxCars 干净四视图对应 `vehicle-track.webp`、`camera-left-front.webp`、`camera-left-rear.webp` 和 `camera-right-rear.webp`，运动模糊素材使用相应的 `camera-*-blur.webp` 文件。
2. ModelNet40 飞机四视图位于 `assets/modelnet40/airplane_0627/clean/`，漂移效果由网页实时生成。
3. 在 `styles.css` 中替换 `.view-1` 至 `.view-4` 的背景图片，或在后续版本中由 `demo_run.json` 动态驱动。
4. 在 `app.js` 中用真实时隙数据替换 `stages`，保留现有字段结构。
5. 最终宣传片位于 `assets/edgecloud-project-film.mp4`，更新时保持相同文件名即可。
6. 东南大学校徽位于 `assets/seu-logo.jpg`，更新时保持相同文件名即可。

当前图片和演示数值均用于视觉与交互验证；导出的 JSON 会保留占位数据说明，避免与正式实验结果混淆。
