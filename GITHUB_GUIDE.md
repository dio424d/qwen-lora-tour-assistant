
# GitHub 上传指南

## 步骤 1：在 GitHub 上创建仓库

1. 访问 [GitHub](https://github.com) 并登录
2. 点击右上角的 "+" 图标，选择 "New repository"
3. 填写仓库信息：
   - Repository name: `qwen-lora-tour-assistant`（或你喜欢的名字）
   - Description: `基于 Qwen1.5-1.8B 的旅游咨询助手，支持 OpenWebUI 接入`
   - 选择 Public（公开）或 Private（私有）
   - **不要**勾选 "Initialize this repository with a README"
   - 点击 "Create repository"

## 步骤 2：关联本地仓库并推送

在你的项目目录下执行以下命令：

```bash
# 添加远程仓库（替换为你的用户名和仓库名）
git remote add origin https://github.com/你的用户名/qwen-lora-tour-assistant.git

# 推送到 GitHub
git push -u origin master
```

如果使用 SSH：

```bash
git remote add origin git@github.com:你的用户名/qwen-lora-tour-assistant.git
git push -u origin master
```

## 步骤 3：验证上传成功

1. 刷新你的 GitHub 仓库页面
2. 你应该能看到所有代码文件
3. 检查 .gitignore 是否正确工作（大模型文件和虚拟环境应该没有被上传）

## 常见问题

### 如果遇到认证问题

使用 Personal Access Token（推荐）：
1. 在 GitHub Settings -> Developer settings -> Personal access tokens 生成 token
2. 推送时使用 token 作为密码

或者使用 GitHub CLI：
```bash
gh auth login
```

### 如果需要修改仓库地址

```bash
git remote set-url origin https://github.com/你的用户名/新仓库名.git
```

### 后续更新代码

```bash
# 添加修改的文件
git add .

# 提交
git commit -m "描述你的修改"

# 推送到 GitHub
git push
```

## 项目结构说明

已上传的文件：
- `README.md` - 项目文档
- `requirements.txt` - 依赖包列表
- `model.py` - 训练代码
- `simple_api.py` - API 服务（推荐使用）
- `api_server.py` - API 服务（完整版）
- `example_usage.py` - API 使用示例
- `config.py` - 配置文件
- `start_server.bat` - Windows 启动脚本
- `.gitignore` - Git 忽略规则

被忽略的文件（不会上传）：
- 大模型文件夹（Qwen*, qwen*）
- LoRA 权重文件夹（qwen-lora-*）
- 虚拟环境（venv/, venv2/）
- 数据集文件（*.txt, *.csv, *.json）
- 压缩包文件

## 下载模型和权重（其他用户使用时）

其他用户克隆仓库后，需要：

1. 下载 Qwen1.5-1.8B-Chat 基础模型到 `./Qwen1.5-1.8B-Chat/`
2. 或使用自己训练的 LoRA 权重

## 许可证建议

建议添加一个开源许可证，如 MIT、Apache 2.0 或 GPL。在 GitHub 仓库设置中可以添加。

