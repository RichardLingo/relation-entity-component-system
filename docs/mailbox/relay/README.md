# 📬 mailbox/relay — 多项目联动治理

本项目已接入 autodo 多项目联动治理体系。

## 信箱结构

```
docs/mailbox/relay/
  registry.md       ← 本项目的治理身份证
  dependencies.md   ← 依赖了谁（上游清单）
  support.md        ← 被谁依赖（下游清单）
  out-up/           ← 写给上游的信（依赖反馈）
  out-down/         ← 写给下游的信（支持建议）
  in-up/            ← 来自上游的信（上游发来的建议）
  in-down/          ← 来自下游的信（下游发来的反馈）
```

## 本项目的信箱角色

- 上游依赖：见 `dependencies.md`
- 下游客户：见 `support.md`

## 如何使用

### 查信
打开 `in-up/` 和 `in-down/`，看是否有新 `.md` 文件。

### 写信
在对应方向目录创建 `{项目ID}.md`，按模板填写。

### 投递
需要启动 mailbox API 后端执行投递，把 outbox 的信投递到对方 inbox。
启动方式：`cd autodo-app && pnpm run mailbox:api`，或 `cd kit-dependency-governance && uv run kdg-api`。

## AI Agent 指引

AI Agent 请参见 `.ai-instructions.md`。
