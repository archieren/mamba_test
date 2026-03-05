# Claude Code 开发流程完全指南

> 本文档介绍 Claude Code 中三大开发工具的使用方法及其协作模式
>
> **版本：** 1.1
> **更新日期：** 2025-02-25
> **重要更新：** 修正了技能命令的使用方式

---

## 快速开始：如何使用 Superpowers 技能

### ⚠️ 重要说明

**Superpowers 技能不是通过 `/skill` 命令使用的！**

正确的使用方式有以下几种：

#### 方式 1：自然对话（推荐）

直接告诉 Claude 你的需求，Claude 会自动判断并使用合适的技能：

```
✅ 你好，我想探索一下用户认证系统的设计方案
→ Claude 自动使用 brainstorming 技能

✅ 帮我调试这个登录 bug
→ Claude 自动使用 systematic-debugging 技能
```

#### 方式 2：明确指定技能

```
✅ 使用 brainstorming 技能探索支付系统的设计方案
✅ 使用 test-driven-development 技能实现用户登录
✅ 使用 verification-before-completion 技能检查代码质量
```

#### 方式 3：斜杠命令（如果配置支持）

某些技能可能支持斜杠命令快捷方式：

```
/plan                     # 启动 Planning with Files
/ralph-loop "任务描述"    # 启动 Ralph 循环
/cancel-ralph             # 取消 Ralph 循环
```

---

## 目录

- [1. 概述](#1-概述)
- [2. Ralph Wiggum 技术](#2-ralph-wiggum-技术)
- [3. Planning with Files](#3-planning-with-files)
- [4. Superpowers 技能系统](#4-superpowers-技能系统)
- [5. 三者的协作模式](#5-三者的协作模式)
- [6. 选择指南](#6-选择指南)
- [7. 实战案例](#7-实战案例)
- [8. 最佳实践](#8-最佳实践)
- [9. 快速参考](#9-快速参考)
- [10. 总结](#10-总结)

---

## 1. 概述

### 三大工具的定位

```
┌─────────────────────────────────────────────────────────┐
│                    Claude Code 开发工具箱                │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Planning with Files    →  地图（文档记录与追踪）        │
│  Superpowers           →  驾驶技能（工作流与最佳实践）  │
│  Ralph Wiggum          →  自动驾驶（自动循环迭代）      │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### 核心概念对比

| 维度 | Planning with Files | Superpowers | Ralph Wiggum |
|------|---------------------|-------------|--------------|
| **性质** | 文档系统 | 技能/工作流系统 | 自动循环引擎 |
| **输出** | md 文档（计划、发现、进度） | 遵循最佳实践的代码 | 迭代完成的工作成果 |
| **重点** | 记录、追踪、持久化 | 流程、方法、规范 | 自动化、持续改进 |
| **适用范围** | 长期、复杂项目 | 任何开发任务 | 有明确完成标准的任务 |
| **会话边界** | 跨会话持久化 | 通常单会话 | 单会话循环 |
| **独立性** | 可独立使用 | 可独立使用 | 通常配合其他工具使用 |

---

## 2. Ralph Wiggum 技术

### 2.1 核心概念

Ralph Wiggum 是一个**迭代开发循环**技术，通过重复执行同一任务来实现持续改进。

**核心思想：**
```bash
while :; do
  cat PROMPT.md | claude-code --continue
done
```

**"自参考"机制：**
- Claude 不是直接看到自己的输出
- 而是通过**修改后的文件**、**git 历史**和**进度文档**看到之前的工作
- 基于这些"记忆"继续改进

### 2.2 工作流程

```
┌──────────────────────────────────────────────────────────┐
│  第 1 次迭代                                              │
│  ├─ Claude 接收任务                                      │
│  ├─ 分析需求，开始实现                                   │
│  ├─ 修改文件，提交代码                                   │
│  └─ 尝试退出 → Stop Hook 拦截                           │
└──────────────────────────────────────────────────────────┘
                           ↓
┌──────────────────────────────────────────────────────────┐
│  第 2 次迭代                                              │
│  ├─ Claude 再次接收**同样的任务**                        │
│  ├─ 看到之前的修改（文件已变更）                         │
│  ├─ 继续改进，或修正问题                                 │
│  └─ 尝试退出 → Stop Hook 拦截                           │
└──────────────────────────────────────────────────────────┘
                           ↓
                        ... 循环 ...
                           ↓
┌──────────────────────────────────────────────────────────┐
│  最后一次迭代                                             │
│  ├─ Claude 检查完成标准                                   │
│  ├─ 输出 <promise>完成标志</promise>                    │
│  └─ Stop Hook 检测到 promise → 真正退出                │
└──────────────────────────────────────────────────────────┘
```

### 2.3 基本用法

#### 启动命令

```bash
/ralph-loop "<任务描述>" --completion-promise "<完成标志>" --max-iterations <数字>
```

#### 参数说明

| 参数 | 说明 | 必需 | 示例 |
|------|------|------|------|
| `任务描述` | 告诉 Claude 要做什么 | 是 | "重构认证模块" |
| `--completion-promise` | 完成标志短语 | 否 | "TASK_COMPLETE" |
| `--max-iterations` | 最大迭代次数 | 否（强烈推荐） | 20 |

#### 完成标志（Completion Promise）

Claude 必须输出特定格式的标签来标志完成：

```
<promise>TASK_COMPLETE</promise>
```

**建议：**
- ✅ 好的标志：`<promise>ALL_TESTS_PASSING</promise>`
- ✅ 好的标志：`<promise>FEATURE_SHIPPING_READY</promise>`
- ❌ 不好的标志：`<promise>完成</promise>` （太普通，可能误触发）

### 2.4 状态文件

Ralph 循环会创建状态文件：`.claude/.ralph-loop.local.md`

```markdown
# Ralph Loop State
Started: 2025-02-25 10:00:00
Iteration: 5
Max Iterations: 50
Completion Promise: TASK_COMPLETE
Prompt: 执行 task_plan.md 中的计划...
```

### 2.5 取消循环

```bash
/cancel-ralph
```

或手动删除：
```bash
rm .claude/.ralph-loop.local.md
```

### 2.6 使用场景

**✅ 适合：**
- 有明确成功标准的任务
- 需要多次迭代和改进的任务
- 可以自动验证完成的任务（如测试通过）
- Greenfield 项目

**❌ 不适合：**
- 需要人工判断或设计决策的任务
- 一次性操作
- 成功标准不明确的任务
- 调试生产问题（应使用 systematic-debugging）

### 2.7 高级技巧

#### 设置合理的 max-iterations

```
小任务:  10-20 次迭代
中任务:  30-50 次迭代
大任务:  100-200 次迭代

经验公式：步骤数 × 2-3 倍
```

#### 在提示中包含检查点

```
/ralph-loop "实施认证系统重构：
1. 设计新架构（检查点：记录设计到 findings.md）
2. 实现核心功能（检查点：运行单元测试）
3. 迁移数据（检查点：验证数据完整性）
4. 更新接口（检查点：集成测试）
5. 所有测试通过时输出 <promise>REFACTOR_DONE</promise>" \
  --max-iterations 50 --completion-promise "REFACTOR_DONE"
```

#### 使用 git 分支保护

```bash
# 启动前
git checkout -b ralph-feature-branch

# 完成后
git diff main  # 查看所有变更
git merge --squash  # 合并所有迭代为一个提交
```

---

## 3. Planning with Files

### 3.1 核心概念

基于文档的规划系统，创建三个核心文档来管理复杂项目。

### 3.2 启动规划

```bash
/plan
# 或
/planning-with-files:plan
```

### 3.3 文档结构

```
.claude/
├── task_plan.md       # 主计划文档
├── findings.md        # 调研发现和笔记
└── progress.md        # 进度追踪
```

### 3.4 文档内容详解

#### task_plan.md - 主计划文档

```markdown
# 任务：重构认证系统

## 目标
将现有的 JWT 认证重构为基于 OAuth2 的认证

## 背景
- 当前使用 Flask-JWT-Extended
- 需要支持第三方登录
- 性能要求：< 100ms

## 步骤
- [ ] 1. 分析现有认证代码
- [ ] 2. 设计 OAuth2 架构
- [ ] 3. 实现 token 端点
- [ ] 4. 实现授权端点
- [ ] 5. 迁移用户数据
- [ ] 6. 更新 API 接口
- [ ] 7. 编写测试
- [ ] 8. 更新文档

## 验收标准
- [ ] 所有现有测试通过
- [ ] 新增测试覆盖率 > 80%
- [ ] 性能无明显下降
- [ ] 文档完整更新
```

#### findings.md - 调研发现

```markdown
# 调研发现

## 现有架构分析
### 目录结构
```
auth/
├── jwt_handler.py
├── middleware.py
└── routes.py
```

### 技术栈
- Flask-JWT-Extended
- Redis (token 存储)
- SQLAlchemy (用户数据)

### 已知问题
1. Token 刷新机制不完善
2. 缺少日志记录
3. ...

## 技术选型
### OAuth2 库选择
候选：authlib vs authcode

选择：authlib
理由：
- 文档完善
- 社区活跃
- 支持 Flask

### 数据存储方案
- 继续使用 Redis
- 增加缓存过期策略
...
```

#### progress.md - 进度追踪

```markdown
# 进度追踪

## 项目信息
- 开始日期：2025-02-20
- 预计完成：2025-02-27
- 当前状态：进行中

## 已完成
- [x] 1. 分析现有认证代码 (2025-02-20)
  - 识别出 3 个核心模块
  - 发现 5 个潜在问题

## 进行中
- [ ] 2. 设计 OAuth2 架构 (开始于 2025-02-21)
  - [ ] 完成数据流设计
  - [ ] 完成接口设计

## 待办
- [ ] 3. 实现 token 端点
- [ ] 4. 实现授权端点
- [ ] 5. 迁移用户数据
- [ ] 6. 更新 API 接口
- [ ] 7. 编写测试
- [ ] 8. 更新文档

## 阻塞问题
无

## 下一步
完成 OAuth2 架构设计的接口设计部分
```

### 3.5 查看进度

```bash
/planning-with-files:status
```

输出：
```
┌─────────────────────────────────────────┐
│  项目进度概览                            │
├─────────────────────────────────────────┤
│  阶段 1: ████████████████████ 100%     │
│  阶段 2: ████████░░░░░░░░░░░░  40%     │
│  阶段 3: ░░░░░░░░░░░░░░░░░░░░   0%     │
├─────────────────────────────────────────┤
│  总体进度：47% (8/17 步骤)             │
└─────────────────────────────────────────┘
```

### 3.6 使用场景

**✅ 最适合：**
- 大型重构项目
- 需要详细调研的任务
- 多天、多会话的开发工作
- 需要保留发现和决策记录
- 团队协作项目

**⚠️ 不太适合：**
- 简单的 bug 修复
- 小型功能添加
- 快速原型开发

---

## 4. Superpowers 技能系统

### 4.1 核心概念

基于**工作流和最佳实践**的技能系统，每个技能针对开发的特定阶段或问题类型。

### 4.2 可用技能列表

#### 流程类技能

| 技能 | 用途 | 阶段 |
|------|------|------|
| `brainstorming` | 创意生成、需求探索 | 开始前 |
| `writing-plans` | 将需求转化为实施计划 | 规划 |
| `executing-plans` | 执行已有的实施计划 | 执行 |
| `subagent-driven-development` | 使用子代理并行开发 | 执行 |
| `dispatching-parallel-agents` | 并行任务调度 | 执行 |
| `finishing-a-development-branch` | 完成开发分支的收尾工作 | 完成阶段 |

#### 质量类技能

| 技能 | 用途 | 阶段 |
|------|------|------|
| `test-driven-development` | TDD 实践 | 开发 |
| `systematic-debugging` | 系统化调试 | 问题解决 |
| `verification-before-completion` | 完成前验证 | 质量保证 |
| `requesting-code-review` | 请求代码审查 | 质量保证 |
| `receiving-code-review` | 接收代码审查反馈 | 质量保证 |

#### 工具类技能

| 技能 | 用途 |
|------|------|
| `using-git-worktrees` | 使用 git worktree 进行隔离开发 |
| `using-superpowers` | 了解如何使用技能系统 |

### 4.3 技能使用流程

```
启动技能 → Claude 遵循技能定义的流程 → 完成特定阶段的工作
```

#### 使用方式

**方式 1：通过对话触发（推荐）**

直接告诉 Claude 你想做什么，Claude 会自动判断是否需要使用技能：

```
你：我想探索一下这个功能的设计方案
→ Claude 自动使用 brainstorming 技能

你：帮我调试这个 bug
→ Claude 自动使用 systematic-debugging 技能
```

**方式 2：明确请求使用技能**

```
你：使用 brainstorming 技能探索支付系统的设计方案
你：使用 test-driven-development 技能实现用户认证
```

**方式 3：使用斜杠命令（如果支持）**

```
/brainstorming 支付系统的设计方案
/test-driven-development 实现用户认证
```

#### 示例：使用 brainstorming 技能

```
你：使用 brainstorming 技能探索用户认证系统的设计方案
```

Claude 会：
1. 提出探索性问题
2. 分析多个可能方案
3. 讨论权衡和选择
4. 最终确定方向

#### 示例：使用 test-driven-development 技能

```
你：使用 test-driven-development 技能实现用户登录功能
```

Claude 会严格遵循 TDD 流程：
1. 先写失败的测试
2. 实现最小代码让测试通过
3. 重构代码
4. 重复上述步骤

### 4.4 使用场景

**每个技能都针对特定场景：**

- **开始新项目** → `brainstorming`
- **制定实施计划** → `writing-plans`
- **开发新功能** → `test-driven-development`
- **遇到 bug** → `systematic-debugging`
- **需要并行处理** → `dispatching-parallel-agents`
- **完成开发** → `verification-before-completion`
- **提交代码** → `requesting-code-review`

### 4.5 技能优先级

当多个技能可能适用时，按以下顺序选择：

```
1. 流程技能优先 (brainstorming, debugging)
   ↓ 确定如何处理任务
2. 实现技能其次 (TDD, patterns)
   ↓ 具体执行
```

示例："让我们构建 X"
```
正确: brainstorming → writing-plans → implementing
错误: 直接开始写代码
```

---

## 5. 三者的协作模式

### 5.1 组合概览

```
┌────────────────────────────────────────────────────────┐
│              三种协作模式                               │
├────────────────────────────────────────────────────────┤
│                                                         │
│  模式 1: Planning + Ralph                              │
│         文档驱动的自动迭代                              │
│                                                         │
│  模式 2: Superpowers + Ralph                           │
│         技能驱动的自动迭代                              │
│                                                         │
│  模式 3: 三者结合                                       │
│         完整的项目工作流                                │
│                                                         │
└────────────────────────────────────────────────────────┘
```

### 5.2 模式 1: Planning with Files + Ralph

**特点：** 文档驱动的自动迭代

```
┌──────────────────────────────────────────┐
│  Planning with Files                     │
│  ├─ task_plan.md (详细计划)             │
│  ├─ findings.md (调研记录)              │
│  └─ progress.md (进度追踪)              │
└──────────────────────────────────────────┘
               ↓
┌──────────────────────────────────────────┐
│  Ralph Wiggum Loop                      │
│  ├─ 读取计划文档                         │
│  ├─ 执行下一步                           │
│  ├─ 更新进度                             │
│  └─ 循环直到完成                         │
└──────────────────────────────────────────┘
```

**使用场景：**
- 大型重构（如：迁移整个模块）
- 多步骤功能开发（如：实现完整的认证系统）
- 需要详细记录的项目

**完整示例：**

```bash
# 第 1 步：创建计划
/plan

# 然后详细描述任务，Claude 会创建三个文档

# 第 2 步：启动 Ralph 循环执行计划
/ralph-loop "执行 .claude/task_plan.md 中定义的任务。
每次迭代：
1. 读取 progress.md 了解当前进度
2. 继续下一个未完成的步骤
3. 完成后更新 progress.md
4. 运行相关测试验证

当 task_plan.md 中所有步骤都标记为完成且所有测试通过时，
输出 <promise>PLAN_COMPLETE</promise>" \
  --completion-promise "PLAN_COMPLETE" \
  --max-iterations 50
```

**迭代过程示例：**

```
迭代 1:
- 读取 task_plan.md → 看到完整计划
- 读取 progress.md → 看到从步骤1开始
- 执行步骤1：分析现有代码
- 更新 progress.md 标记步骤1完成
- 尝试退出 → 被拦截

迭代 2:
- 再次收到同样提示
- 读取 progress.md → 看到步骤1完成
- 继续步骤2：设计新架构
- 更新 findings.md 记录设计决策
- 更新 progress.md 标记步骤2完成
- 尝试退出 → 被拦截

... 继续迭代 ...

迭代 N:
- 读取 progress.md → 所有步骤完成
- 运行所有测试 → 通过
- 输出 <promise>PLAN_COMPLETE</promise>
- 循环结束
```

### 5.3 模式 2: Superpowers + Ralph

**特点：** 技能驱动的自动迭代

```
┌──────────────────────────────────────────┐
│  Superpowers Skill                       │
│  ├─ test-driven-development (TDD流程)   │
│  ├─ systematic-debugging (调试流程)     │
│  └─ verification-before-completion       │
└──────────────────────────────────────────┘
               ↓
┌──────────────────────────────────────────┐
│  Ralph Wiggum Loop                      │
│  ├─ 应用技能的流程                       │
│  ├─ 遵循最佳实践                         │
│  ├─ 自动化质量检查                       │
│  └─ 循环直到满足技能标准                 │
└──────────────────────────────────────────┘
```

**使用场景：**
- 需要严格流程控制的任务
- 需要特定开发方法（如 TDD）
- 需要系统化调试或验证

**示例 1: TDD + Ralph**

```bash
/ralph-loop "严格遵循 test-driven-development 技能的流程：

每次迭代：
1. 检查 findings.md 了解当前状态
2. 为下一个功能点编写失败的测试
3. 实现最小代码让测试通过
4. 重构优化代码
5. 运行完整测试套件验证
6. 更新 findings.md 记录决策和问题

当所有功能实现、测试通过、覆盖率 > 80% 时，
输出 <promise>TDD_COMPLETE</promise>" \
  --completion-promise "TDD_COMPLETE" \
  --max-iterations 30
```

**示例 2: 系统化调试 + Ralph**

```bash
/ralph-loop "遵循 systematic-debugging 技能的流程调试这个 bug：

Bug 现象：用户登录后 token 无效

每次迭代：
1. 尝试重现问题
2. 收集证据（日志、状态）
3. 分析根本原因
4. 提出并实施修复
5. 验证修复有效
6. 检查是否有类似问题
7. 更新 findings.md 记录发现

当 bug 修复、所有测试通过、回归测试通过时，
输出 <promise>BUG_FIXED</promise>" \
  --completion-promise "BUG_FIXED" \
  --max-iterations 20
```

**示例 3: 验证 + Ralph**

```bash
/ralph-loop "使用 verification-before-completion 技能检查代码：

每次迭代：
1. 运行测试套件
2. 检查代码覆盖率
3. 运行 linter 和类型检查
4. 检查文档完整性
5. 性能基准测试
6. 更新 findings.md 记录问题
7. 修复发现的问题

当所有检查通过、没有遗留问题时，
输出 <promise>VERIFICATION_COMPLETE</promise>" \
  --completion-promise "VERIFICATION_COMPLETE" \
  --max-iterations 15
```

### 5.4 模式 3: 三者结合（完整工作流）

**特点：** 完整的企业级开发流程

```
阶段 1: 需求探索
├─ 使用 brainstorming 技能
└─ 生成需求文档

阶段 2: 制定计划
├─ 使用 writing-plans 技能
│   或使用 Planning with Files
└─ 创建详细计划文档

阶段 3: 自动执行
├─ 选择合适的 superpowers 技能
│   (如 TDD, subagent-driven-development)
└─ 用 Ralph 循环自动迭代

阶段 4: 质量保证
├─ 使用 verification-before-completion
├─ 使用 requesting-code-review
└─ 确保代码质量
```

**完整示例：开发用户认证功能**

```bash
# === 阶段 1: 需求探索 ===
# 对 Claude 说：使用 brainstorming 技能探索用户认证功能的需求

# Claude 会问你：
# - 这个功能的核心需求是什么？
# - 有哪些技术方案可以选择？
# - 各有什么优劣？
# - 最终推荐哪个方案？

# === 阶段 2: 制定计划 ===
# 方案 A: 对 Claude 说：使用 writing-plans 技能制定实施计划

# 方案 B: 使用 Planning with Files
/plan

# === 阶段 3: 自动执行（带 TDD）===
/ralph-loop "使用 test-driven-development 流程实施计划：

每次迭代：
1. 读取 task_plan.md 了解下一步
2. 读取 progress.md 查看进度
3. 为下一个功能点编写测试（TDD）
4. 实现功能让测试通过
5. 重构代码
6. 运行完整测试套件
7. 更新 progress.md
8. 记录决策到 findings.md

当所有功能完成、测试通过、覆盖率 > 80%、
文档完整时输出 <promise>FEATURE_SHIPPABLE</promise>" \
  --completion-promise "FEATURE_SHIPPABLE" \
  --max-iterations 100

# === 阶段 4: 完成前验证 ===
# 对 Claude 说：使用 verification-before-completion 技能检查代码

# Claude 会：
# - 运行完整测试套件
# - 检查代码质量
# - 验证文档完整性
# - 性能测试
# - 安全检查

# === 阶段 5: 代码审查 ===
# 对 Claude 说：使用 requesting-code-review 技能生成 PR

# Claude 会：
# - 生成 Pull Request
# - 创建详细的变更说明
# - 列出审查要点
```

### 5.5 协作模式对比

| 维度 | Planning + Ralph | Superpowers + Ralph | 三者结合 |
|------|------------------|---------------------|----------|
| **复杂度** | 中等 | 中等 | 高 |
| **文档化** | 强 | 中等 | 强 |
| **流程控制** | 弱 | 强 | 强 |
| **适用项目** | 中大型 | 需要特定流程 | 大型/企业级 |
| **学习曲线** | 低 | 中 | 高 |
| **质量保证** | 基础 | 强 | 最强 |

---

## 6. 选择指南

### 6.1 决策树

```
开始
  │
  ├─ 任务类型？
  │   │
  │   ├─ 探索性/不确定性高
  │   │   └─→ 使用 brainstorming 技能
  │   │
  │   ├─ Bug/问题解决
  │   │   └─→ 使用 systematic-debugging 技能
  │   │
  │   ├─ 简单功能（< 100 行代码）
  │   │   └─→ 直接实现 或 单个 Superpower 技能
  │   │
  │   ├─ 中型功能（100-1000 行）
  │   │   │
  │   │   ├─ 需要严格流程？
  │   │   │   └─ 是 → Superpower (TDD) + Ralph
  │   │   │   └─ 否 → Planning + Ralph
  │   │   │
  │   └─ 大型项目（> 1000 行或多模块）
  │       └─→ 三者结合
  │
  └─ 需要自动迭代？
      │
      ├─ 是 → 加入 Ralph 循环
      └─ 否 → 手动执行
```

### 6.2 快速参考表

| 任务类型 | 推荐方案 | 理由 |
|---------|---------|------|
| **快速修复 bug** | systematic-debugging | 直接定位问题 |
| **探索新功能** | brainstorming | 需要创意探索 |
| **简单脚本** | 直接实现 | 不需要复杂流程 |
| **添加测试** | TDD + Ralph | 自动化 TDD 流程 |
| **中型功能** | Planning + Ralph | 需要追踪进度 |
| **需要严格质量** | Superpower + Ralph | 如 TDD/验证 |
| **大型重构** | 三者结合 | 完整工作流 |
| **多天开发** | Planning with Files | 文档持久化 |
| **并行任务** | dispatching-parallel-agents | 提高效率 |

### 6.3 场景示例

#### 场景 1：修复生产问题

```bash
# 对 Claude 说：使用 systematic-debugging 技能调试这个 bug

# 不需要 Ralph，因为需要人工判断
```

#### 场景 2：实现一个 API 端点

```bash
# 方案 A：简单情况
# 直接实现，不需要复杂工具

# 方案 B：需要质量保证
/ralph-loop "使用 test-driven-development 流程实现用户 API..." \
  --completion-promise "API_DONE" --max-iterations 20
```

#### 场景 3：重构整个模块

```bash
# 完整流程
# 对 Claude 说：使用 brainstorming 技能探索重构方案
/plan                          # 制定计划
/ralph-loop "执行计划..."      # 自动执行
# 对 Claude 说：使用 verification-before-completion 技能验证
```

#### 场景 4：实现复杂功能（如支付系统）

```bash
# 三者结合
# 对 Claude 说：使用 brainstorming 技能探索需求
# 对 Claude 说：使用 writing-plans 技能制定计划
/ralph-loop "TDD 流程执行..."         # TDD + Ralph
# 对 Claude 说：使用 verification-before-completion 技能验证
# 对 Claude 说：使用 requesting-code-review 技能生成 PR
```

---

## 7. 实战案例

### 7.1 案例 1：实现用户权限系统

#### 传统方式（低效）

```
你 → Claude: "实现用户权限系统"
Claude → 生成代码
你 → 检查，发现有问题
你 → Claude: "修复这个问题"
Claude → 修复
你 → 发现另一个问题
你 → Claude: "再修复"
...（反复手动交互 10+ 次）
```

#### 使用 Ralph Wiggum（高效）

```bash
/ralph-loop "实现用户权限系统，包括：
1. 角色定义（管理员、普通用户、访客）
2. 权限检查装饰器
3. API 权限控制
4. 单元测试

每次迭代：
1. 实现一个功能点
2. 编写/更新测试
3. 运行测试验证
4. 记录决策到 findings.md

当所有功能实现、测试通过时，
输出 <promise>PERMISSIONS_DONE</promise>" \
  --max-iterations 30 \
  --completion-promise "PERMISSIONS_DONE"
```

**结果：** 自动迭代 15 次后完成，无需人工干预

### 7.2 案例 2：重构认证系统

#### 使用 Planning + Ralph

**第 1 步：创建计划**

```bash
/plan
```

创建 `task_plan.md`：
```markdown
# 重构认证系统

## 目标
从 JWT 迁移到 OAuth2

## 步骤
- [ ] 1. 分析现有 JWT 实现
- [ ] 2. 设计 OAuth2 架构
- [ ] 3. 实现授权服务器
- [ ] 4. 实现资源服务器
- [ ] 5. 数据迁移
- [ ] 6. API 更新
- [ ] 7. 测试
- [ ] 8. 文档

## 验收标准
- 所有测试通过
- 性能无明显下降
- 文档完整
```

**第 2 步：Ralph 循环**

```bash
/ralph-loop "执行 task_plan.md：
1. 读取 progress.md 了解进度
2. 执行下一步
3. 更新 progress.md
4. 运行测试

完成后输出 <promise>AUTH_REFACTOR_COMPLETE</promise>" \
  --max-iterations 50 \
  --completion-promise "AUTH_REFACTOR_COMPLETE"
```

**结果：**
- 清晰的进度追踪
- 可随时恢复
- 系统化执行

### 7.3 案例 3：开发支付功能（高质量要求）

#### 使用三者结合

**阶段 1：头脑风暴**

```bash
# 对 Claude 说：使用 brainstorming 技能探索支付系统的设计方案
```

探索：
- 支付方式选择（支付宝/微信/Stripe）
- 安全性考虑
- 退款流程
- 异步处理

**阶段 2：制定计划**

```bash
# 对 Claude 说：使用 writing-plans 技能制定支付系统实施计划
```

创建详细实施计划

**阶段 3：TDD + Ralph**

```bash
/ralph-loop "使用 test-driven-development 流程：

每次迭代：
1. 编写失败的测试（先测试）
2. 实现功能（后实现）
3. 重构代码
4. 运行测试套件
5. 更新 findings.md

当所有功能完成、测试覆盖率 > 90%、
安全扫描通过时输出 <promise>PAYMENT_SHIPPABLE</promise>" \
  --max-iterations 100 \
  --completion-promise "PAYMENT_SHIPPABLE"
```

**阶段 4：验证**

```bash
# 对 Claude 说：使用 verification-before-completion 技能验证支付系统
```

- 完整测试
- 安全审查
- 性能测试
- 压力测试

**阶段 5：代码审查**

```bash
# 对 Claude 说：使用 requesting-code-review 技能生成 PR
```

生成 PR 和审查清单

**结果：**
- 高质量代码
- 完整测试覆盖
- 安全可靠
- 文档完整

---

## 8. 最佳实践

### 8.1 通用原则

#### 1. 先规划，后执行

```
❌ 错误：直接开始 coding
✅ 正确：brainstorming → planning → executing
```

#### 2. 明确完成标准

```
❌ 错误："完成后退出"
✅ 正确："测试通过且覆盖率 > 80% 时输出 <promise>DONE</promise>"
```

#### 3. 设置安全上限

```
❌ 错误：没有 max-iterations（可能无限循环）
✅ 正确：--max-iterations 50
```

#### 4. 保持文档可读

```
❌ 错误：progress.md 内容混乱
✅ 正确：清晰的进度标记、阻塞问题、下一步
```

#### 5. 使用版本控制

```bash
# 启动前
git checkout -b feature-branch

# 定期提交（可在 Ralph 循环中加入检查点）
# 完成后
git diff main
git merge --squash
```

### 8.2 Planning with Files 最佳实践

#### 计划要具体

```markdown
❌ 错误：
- [ ] 实现功能

✅ 正确：
- [ ] 实现用户登录
  - [ ] 创建登录 API 端点 POST /auth/login
  - [ ] 实现密码验证逻辑
  - [ ] 返回 JWT token
  - [ ] 添加错误处理
  - [ ] 编写单元测试
```

#### findings.md 要详细记录

```markdown
## 技术选型

### OAuth2 库选择
候选：authlib vs authcode

**选择：authlib**

**理由：**
1. 文档完善（有完整的中文文档）
2. 社区活跃（GitHub 2k+ stars）
3. 支持 Flask（直接集成）
4. 维护频繁（最新版本 2024-12）

**风险评估：**
- 低风险：成熟稳定
- 依赖项：需要 cryptography >= 3.4
```

#### progress.md 要反映真实状态

```markdown
## 进行中
- [ ] 2. 设计 OAuth2 架构
  - [x] 数据流设计
  - [x] 接口设计
  - [ ] 安全模型设计（阻塞：等待安全团队审查）

## 阻塞问题
1. 安全模型需要安全团队审查
   - 联系人：张三
   - 预计时间：2025-02-26

## 下一步
等待安全审查通过后继续
```

### 8.3 Ralph Wiggum 最佳实践

#### Promise 要具体且独特

```
❌ 错误：<promise>完成</promise>  # 太普通
❌ 错误：<promise>done</promise>   # 可能误触发

✅ 正确：<promise>FEATURE_SHIPPING_READY</promise>
✅ 正确：<promise>ALL_INTEGRATION_TESTS_PASSING</promise>
```

#### Max-iterations 要合理

```
步骤数：10
推荐：max-iterations = 20-30

理由：
- 每个步骤可能需要 2-3 次迭代
- 预留调试时间
- 避免过早终止
```

#### 提示要结构化

```
❌ 错误：
"实现功能"

✅ 正确：
"实现认证功能：
1. 每次迭代检查 progress.md
2. 继续下一个未完成的步骤
3. 完成后更新 progress.md
4. 运行测试验证
5. 记录决策到 findings.md
当所有步骤完成且测试通过时输出 <promise>DONE</promise>"
```

#### 设置检查点

```
/ralph-loop "实施功能：

检查点 1：实现阶段
- 每个功能实现后运行单元测试

检查点 2：集成阶段
- 所有功能实现后运行集成测试

检查点 3：完成阶段
- 运行完整测试套件
- 检查覆盖率
- 验证文档

所有检查点通过时输出 <promise>DONE</promise>"
```

### 8.4 Superpowers 最佳实践

#### 选择正确的技能

```
任务：修复 bug
❌ 错误：使用 brainstorming（探索性）
✅ 正确：使用 systematic-debugging（针对性）

任务：新功能设计
❌ 错误：使用 systematic-debugging（解决问题）
✅ 正确：使用 brainstorming（探索方案）
```

#### 遵循技能的流程

```
使用 TDD 技能时：
❌ 错误：先写代码，后写测试
✅ 正确：先写测试，后写代码

使用 systematic-debugging 时：
❌ 错误：直接修复问题
✅ 正确：先收集证据，后定位根源
```

#### 组合技能使用

```
开发新功能：
brainstorming → writing-plans → test-driven-development → verification

修复 bug：
systematic-debugging → verification

重构：
brainstorming → writing-plans → executing-plans → verification → code-review
```

### 8.5 三者结合的最佳实践

#### 大型项目的标准流程

```
1. 需求阶段（1-2 天）
   对 Claude 说：使用 brainstorming 技能探索项目需求
   → 生成需求文档

2. 规划阶段（1 天）
   /plan
   → 生成 task_plan.md, findings.md, progress.md

3. 开发阶段（N 天）
   /ralph-loop "执行计划..." --max-iterations 200

   中途检查：
   - 每天早上：/planning-with-files:status
   - 每周：git commit 保存进度

4. 验证阶段（1 天）
   对 Claude 说：使用 verification-before-completion 技能验证代码

5. 审查阶段（1 天）
   对 Claude 说：使用 requesting-code-review 技能生成 PR
```

#### 分阶段实施

```bash
# 第 1 周：核心功能
/ralph-loop "实现核心功能..." --max-iterations 50

# 第 2 周：辅助功能
# 更新 task_plan.md，标记核心功能完成
/ralph-loop "实现辅助功能..." --max-iterations 50

# 第 3 周：优化和测试
# 对 Claude 说：使用 verification-before-completion 技能验证
```

#### 团队协作

```
角色分工：
- 架构师：brainstorming + writing-plans
- 开发者：ralph-loop 执行计划
- QA：verification
- Tech Lead：code-review

文档共享：
- task_plan.md：所有人可见
- findings.md：记录技术决策
- progress.md：追踪团队进度
```

### 8.6 常见问题和解决方案

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| **Ralph 无限循环** | promise 未正确输出 | 设置 max-iterations |
| **迭代方向错误** | 计划不够清晰 | 完善 task_plan.md |
| **重复做同一件事** | progress.md 未更新 | 在提示中强调更新进度 |
| **质量下降** | 缺乏验证标准 | 在计划中加入检查点和测试 |
| **代码丢失** | 迭代过多未提交 | 设置自动提交检查点 |
| **不知道用什么技能** | 不熟悉技能 | 参考 6.2 快速参考表 |
| **文档太复杂** | 过度规划 | 简化计划，聚焦核心 |
| **Ralph 太慢** | 每次迭代做太多 | 分解为更小的步骤 |

### 8.7 性能优化

#### Ralph 循环优化

```bash
# ❌ 慢：每次迭代运行完整测试套件
/ralph-loop "实现功能...每次运行 pytest"

# ✅ 快：阶段性运行测试
/ralph-loop "实现功能...
- 实现阶段：运行单元测试
- 集成阶段：运行集成测试
- 完成阶段：运行完整测试"
```

#### 文档维护

```markdown
# findings.md 过大？

## 方案 1：归档旧内容
findings.md（当前）
findings-archive.md（历史）

## 方案 2：分类记录
findings.md
├─ decisions.md（决策）
├─ research.md（调研）
└─ issues.md（问题）
```

---

## 9. 快速参考

### 9.1 命令速查表

| 命令 | 用途 |
|------|------|
| `/plan` | 启动 Planning with Files |
| `/planning-with-files:status` | 查看计划进度 |
| `/ralph-loop "..."` | 启动 Ralph 循环 |
| `/cancel-ralph` | 取消 Ralph 循环 |
| **对 Claude 说** | 使用 Superpowers 技能 |

**Superpowers 技能使用方式：**
- 直接对话：对 Claude 说"使用 brainstorming 技能..."
- 自然触发：告诉 Claude 你的需求，自动判断使用哪个技能

### 9.2 常用技能列表

| 技能 | 简写 | 用途 |
|------|------|------|
| `brainstorming` | 探索 | 需求探索、方案选择 |
| `writing-plans` | 计划 | 制定实施计划 |
| `test-driven-development` | TDD | 测试驱动开发 |
| `systematic-debugging` | 调试 | 系统化调试 |
| `verification-before-completion` | 验证 | 完成前验证 |
| `requesting-code-review` | 审查 | 代码审查 |

### 9.3 典型组合模式

```
# 简单任务
直接实现

# 需要质量
TDD + Ralph

# 中型任务
Planning + Ralph

# 复杂任务
brainstorming → planning → TDD + Ralph → verification → review

# 调试问题
systematic-debugging

# 并行任务
dispatching-parallel-agents
```

---

## 10. 总结

### 核心要点

1. **Planning with Files** = 地图
   - 提供方向和追踪
   - 适合长期项目

2. **Superpowers** = 驾驶技能
   - 提供流程和最佳实践
   - 适合特定阶段

3. **Ralph Wiggum** = 自动驾驶
   - 提供自动化和持续改进
   - 适合有明确标准的任务

### 选择原则

- **小任务**：直接实现或单个技能
- **中任务**：Planning + Ralph 或 Superpower + Ralph
- **大任务**：三者结合
- **质量敏感**：Superpowers (TDD/verification) + Ralph

### 成功关键

1. 明确的完成标准
2. 合理的安全上限（max-iterations）
3. 清晰的文档和进度追踪
4. 适当的检查点和验证
5. 版本控制和备份

---

## 附录

### A. 相关资源

- **Ralph Wiggum 原始技术：** https://ghuntley.com/ralph/
- **Ralph Orchestrator：** https://github.com/mikeyobrien/ralph-orchestrator
- **Claude Code 文档：** [官方文档]

### B. 更新日志

| 版本 | 日期 | 更新内容 |
|------|------|----------|
| 1.0 | 2025-02-25 | 初始版本 |

### C. 反馈和贡献

如有问题或建议，请通过以下方式反馈：
- 项目 Issues
- 文档 Pull Request

---

**文档结束**

*本文档由 Claude Code 自动生成，记录了 Ralph Wiggum、Superpowers 和 Planning with Files 三大工具的使用方法和协作模式。*
