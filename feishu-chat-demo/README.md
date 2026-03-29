# 飞书聊天消息跳转历史消息 Demo

一个模拟飞书聊天消息列表的最小可演示 Demo，实现了跳转历史消息、分段加载与合并、平滑滚动等核心功能。

## 运行方式

```bash
cd feishu-chat-demo
npm install
npm run dev
```

## 技术栈

- React 19 + TypeScript
- Vite 8

---

## 项目结构

```
src/
├── types.ts            # 核心类型定义
├── mockData.ts         # Mock 数据层，模拟 2000 条消息和 4 个 API
├── useMessageStore.ts  # 消息段状态管理 Hook（加载、合并、跳转）
├── ChatList.tsx        # 消息列表组件（滚动锚点保持、平滑动画、无限滚动）
├── Sidebar.tsx         # 侧边栏组件（跳转输入框、快捷按钮、功能说明）
├── App.tsx             # 主布局组件（顶部栏 + 聊天区 + 侧边栏）
└── main.tsx            # 应用入口 + 全局样式注入
```

---

## 核心数据模型

### `types.ts` — 三个核心类型

```typescript
interface Message {
  id: number;          // 消息唯一标识，1~2000 连续递增
  content: string;     // 消息文本内容
  senderId: string;    // 发送者 ID（'user1'~'user4' 和 'me'）
  senderName: string;  // 发送者名称
  avatar: string;      // 发送者头像（Emoji 表情）
  timestamp: number;   // 消息时间戳（毫秒），从 2026-01-01 起每条间隔 1 分钟
}

interface Segment {
  id: string;              // 消息段唯一 ID，格式 'seg_1', 'seg_2', ...
  messages: Message[];     // 该段包含的消息数组，按 id 升序排列
  hasMoreBefore: boolean;  // 该段之前是否还有更早的消息可加载
  hasMoreAfter: boolean;   // 该段之后是否还有更新的消息可加载
}

interface FetchResult {
  messages: Message[];     // 本次请求返回的消息数组
  hasMoreBefore: boolean;  // 返回区间之前是否还有更多消息
  hasMoreAfter: boolean;   // 返回区间之后是否还有更多消息
}
```

**设计要点：**

- **Segment（消息段）** 是整个架构的核心抽象。每次加载操作（首屏加载、向上/向下滚动加载、跳转加载）都会产生或修改一个 Segment。多个 Segment 可以独立存在（中间有未加载的 gap），也可以在数据重叠时自动合并。
- `hasMoreBefore` / `hasMoreAfter` 控制该段两端是否还能继续加载，用于驱动无限滚动的触发逻辑和 UI 提示。

---

## Mock 数据层

### `mockData.ts` — 数据生成 + 4 个模拟 API

#### 数据生成

- 预先生成 `TOTAL_MESSAGES = 2000` 条消息，存储在内存数组 `allMessages` 中
- 5 个发送者轮流发消息（`i % 5`），其中 `senderId === 'me'` 的消息在 UI 中右对齐显示
- 20 个内容模板轮流使用（`i % 20`），每条消息前缀 `[#id]` 方便识别
- 时间戳从 `2026-01-01 00:00` 起，每条消息间隔 1 分钟
- 所有 API 通过 `delay()` 模拟网络延迟（300~500ms）

#### 四个模拟 API

| API 函数 | 延迟 | 用途 | 返回逻辑 |
|---------|------|------|---------|
| `fetchLatestMessages()` | 300ms | 首屏加载 | 返回最后 30 条消息（id 1971~2000），`hasMoreBefore=true, hasMoreAfter=false` |
| `fetchMessagesBefore(messageId)` | 400ms | 向上滚动加载更早消息 | 找到 messageId 在数组中的位置 `idx`，返回 `[idx-30, idx)` 区间的消息 |
| `fetchMessagesAfter(messageId)` | 400ms | 向下滚动加载更新消息 | 找到 messageId 在数组中的位置 `idx`，返回 `(idx, idx+30]` 区间的消息 |
| `fetchMessagesAround(messageId)` | 500ms | 跳转历史消息 | 找到 messageId 在数组中的位置 `idx`，返回 `[idx-30, idx+30]` 区间的消息（最多 61 条） |

每个 API 返回的 `FetchResult` 都正确计算 `hasMoreBefore` 和 `hasMoreAfter`，例如当 `start === 0` 时 `hasMoreBefore = false`，表示已经到达消息列表的最顶端。

#### 快捷跳转书签

`getSearchableMessages()` 返回 11 个预设的跳转目标，覆盖消息列表的各个区域：

```
#1, #50, #100, #200, #500, #800, #1000, #1200, #1500, #1700, #1950
```

---

## 状态管理

### `useMessageStore.ts` — 消息段的增删改查

这是整个应用的状态中枢，管理 `Segment[]` 数组和加载/跳转状态。

#### 状态定义

```typescript
const [segments, setSegments] = useState<Segment[]>([]);    // 所有消息段
const [loading, setLoading] = useState(false);               // 全局加载状态
const [jumpingToId, setJumpingToId] = useState<number | null>(null);  // 当前跳转目标
const loadingRef = useRef(false);       // 防止并发请求的互斥锁
const segmentsRef = useRef<Segment[]>([]); // segments 的同步快照，避免在 async 中读到旧闭包
```

**`segmentsRef` 的作用：**
React 的 `useState` 在异步回调中可能读到旧的闭包值。`segmentsRef` 通过 `updateSegments` 封装保持与 `segments` 状态同步，使 `loadBefore`/`loadAfter`/`jumpToMessage` 等 async 函数在 `await` 之后仍能读到最新的段数据。

```typescript
const updateSegments = useCallback((updater: (prev: Segment[]) => Segment[]) => {
  setSegments(prev => {
    const next = updater(prev);
    segmentsRef.current = next;  // 同步更新 ref
    return next;
  });
}, []);
```

#### `loadingRef` 互斥锁

所有加载函数在入口处检查 `loadingRef.current`，防止用户快速操作或滚动触发多次并发请求：

```typescript
if (loadingRef.current) return;  // 有请求正在进行，直接返回
loadingRef.current = true;       // 占锁
setLoading(true);
try {
  // ... 异步操作
} finally {
  setLoading(false);
  loadingRef.current = false;    // 释放锁
}
```

#### 核心操作

##### 1. `loadInitial()` — 首屏加载

```
调用 fetchLatestMessages() → 获取最后 30 条消息 → 创建第一个 Segment → 设置到 segments
```

- 进入会话时调用一次，在 `App.tsx` 的 `useEffect` 中触发
- 创建的 Segment：`{ id: 'seg_1', messages: [#1971...#2000], hasMoreBefore: true, hasMoreAfter: false }`

##### 2. `loadBefore(segmentId)` — 向上加载更早消息

```
从 segmentsRef 中找到目标 Segment
→ 取其第一条消息的 id 作为锚点
→ 调用 fetchMessagesBefore(firstMsgId) 获取前 30 条
→ 将新消息 prepend 到该 Segment 的 messages 数组前面
→ 更新 hasMoreBefore
→ 调用 mergeSegments() 检查是否需要与相邻段合并
```

##### 3. `loadAfter(segmentId)` — 向下加载更新消息

```
从 segmentsRef 中找到目标 Segment
→ 取其最后一条消息的 id 作为锚点
→ 调用 fetchMessagesAfter(lastMsgId) 获取后 30 条
→ 将新消息 append 到该 Segment 的 messages 数组后面
→ 更新 hasMoreAfter
→ 调用 mergeSegments() 检查是否需要与相邻段合并
```

##### 4. `jumpToMessage(messageId)` — 跳转历史消息

```
第一步：检查 messageId 是否已在某个 Segment 中
  → 如果已加载：直接设置 jumpingToId，触发 ChatList 的滚动动画
  → 如果未加载：继续第二步

第二步：调用 fetchMessagesAround(messageId) 获取前后各 30 条消息
  → 创建新的 Segment
  → 调用 mergeSegments([...现有段, 新段])
  → 设置 jumpingToId，触发 ChatList 的滚动动画
```

#### 消息段合并算法 `mergeSegments()`

这是保证消息列表连续性的关键算法。当用户跳转到某个历史位置后不断滚动加载，新加载的消息可能会与其他已有的 Segment 产生重叠或变得相邻，此时需要合并。

**算法步骤：**

```
1. 将所有 Segment 按照首条消息的 id 升序排序
2. 初始化 merged = [第一个段]
3. 遍历剩余段，对每个 curr：
   a. 取 merged 最后一个段 prev
   b. 比较 prev.最后一条消息.id 和 curr.第一条消息.id
   c. 如果 currFirst <= prevLast + 1（重叠或相邻）：
      - 用 Map<id, Message> 去重合并两段的所有消息
      - 合并后按 id 排序
      - hasMoreBefore = prev.hasMoreBefore（取前段的前边界）
      - hasMoreAfter = curr.hasMoreAfter（取后段的后边界）
      - 保留 prev 的 segment id
   d. 如果 currFirst > prevLast + 1（有 gap）：
      - curr 作为独立段加入 merged
4. 返回 merged
```

**合并判定条件 `currFirst <= prevLast + 1`：**

- `currFirst <= prevLast`：两段有消息 ID 重叠，需要去重合并
- `currFirst === prevLast + 1`：两段刚好首尾相接（相邻），消息是连续的，合并后用户看到的是一个无缝列表

**合并时的去重策略：**

使用 `Map<number, Message>` 以消息 ID 为 key，先放入 prev 的所有消息，再放入 curr 的消息（后者覆盖前者），最后按 id 排序。这确保了不会出现重复消息。

**边界标记的继承：**

合并后的段保留前段（prev）的 `hasMoreBefore`（因为它在更早的位置），以及后段（curr）的 `hasMoreAfter`（因为它在更晚的位置），确保合并后的段两端还能继续加载。

---

## 消息列表组件

### `ChatList.tsx` — 最复杂的组件

负责渲染消息列表、管理滚动锚点保持、处理跳转动画、驱动无限滚动加载。

#### 一、滚动锚点保持机制

**问题：** 当用户滚动到列表顶部触发 loadBefore 后，新消息被 prepend 到列表顶部。DOM 更新后 `scrollTop` 不变，但内容整体下移，用户视角中原来看到的消息会突然跳到下方——这就是"滚动条跳动"。

**解决方案：三步锚点保持**

```
步骤 1 — 持续快照锚点（useEffect，每次 paint 后执行）
  ↓
  遍历所有 [data-msg-id] 元素，找到第一个底边 > scrollTop 的元素
  记录 { msgId: 消息ID, offset: 元素top - scrollTop }
  ↓
步骤 2 — 检测变化（useLayoutEffect，DOM 更新后、浏览器绘制前执行）
  ↓
  比较 currentFirstId 与 prevFirstIdRef：首条消息 ID 变了，说明有 prepend
  ↓
步骤 3 — 恢复位置
  ↓
  找到锚点消息的新 DOM 元素
  计算新的 scrollTop = anchorEl.offsetTop - anchor.offset
  直接设置 container.scrollTop（在浏览器绘制前完成，用户无感知）
```

**关键代码：**

```typescript
// 步骤 1：每次 paint 后快照当前锚点
useEffect(() => {
  if (!isAnimatingRef.current) {
    anchorRef.current = getFirstVisibleAnchor();
  }
});

// 步骤 2+3：检测 prepend 并恢复
useLayoutEffect(() => {
  if (currentFirstId === prevFirstIdRef.current) return; // 没变化
  const wasNull = prevFirstIdRef.current === null;
  prevFirstIdRef.current = currentFirstId;
  if (wasNull || isAnimatingRef.current || jumpingToId !== null) return; // 跳过初始化和动画中

  const container = containerRef.current;
  const anchor = anchorRef.current;
  if (!container || !anchor) return;

  const anchorEl = container.querySelector(`[data-msg-id="${anchor.msgId}"]`);
  if (!anchorEl) return;
  container.scrollTop = anchorEl.offsetTop - anchor.offset; // 恢复到锚点位置
}, [currentFirstId, jumpingToId]);
```

**为什么用 `useLayoutEffect` 而不是 `useEffect`？**

- `useLayoutEffect` 在 DOM 变更后、浏览器绘制（paint）前同步执行
- 此时修改 `scrollTop` 用户看不到任何闪烁，实现了真正的"无感知"位置恢复
- 如果用 `useEffect`（异步，在 paint 后执行），用户会先看到跳动再看到恢复，产生闪烁

**为什么对 prepend 和 append 分别处理？**

- **Prepend**（向上加载更早消息）：`currentFirstId` 变化时触发，需要恢复锚点因为新内容加到了上方
- **Append**（向下加载更新消息）：`currentLastId` 变化时触发，虽然 append 通常不会改变当前视口位置，但在消息段合并等复杂场景下仍可能需要锚点恢复

**跳过条件：**

- `wasNull`（首次渲染）：初始加载走的是"滚动到底部"逻辑，不需要锚点恢复
- `isAnimatingRef.current`（动画中）：跳转动画正在控制 scrollTop，不应干扰
- `jumpingToId !== null`（有跳转请求）：跳转逻辑会接管滚动位置

#### 二、跳转滚动动画

当 `jumpingToId` 变化时触发：

```typescript
useEffect(() => {
  if (jumpingToId === null) return;

  const targetEl = container.querySelector(`[data-msg-id="${jumpingToId}"]`);
  if (!targetEl) return;

  isAnimatingRef.current = true;  // 标记动画中，阻止锚点恢复和滚动加载

  // 计算目标位置：让消息出现在容器垂直中心
  const targetCenter = targetEl.offsetTop - containerHeight / 2 + targetEl.offsetHeight / 2;

  smoothScrollTo(container, Math.max(0, targetCenter), 600).then(() => {
    isAnimatingRef.current = false;
    setHighlightId(jumpingToId);     // 高亮目标消息
    onClearJump();                    // 清除跳转状态
    anchorRef.current = getFirstVisibleAnchor();  // 重新快照锚点
    setTimeout(() => setHighlightId(null), 2000);  // 2秒后取消高亮
  });
}, [jumpingToId, allMessages.length, onClearJump, getFirstVisibleAnchor]);
```

**`smoothScrollTo()` 函数：**

```typescript
function smoothScrollTo(element, target, duration = 600ms) {
  // 使用 requestAnimationFrame 驱动的自定义动画
  // 缓动函数：easeInOutCubic
  //   t < 0.5: 4t³ (加速段)
  //   t ≥ 0.5: 1 - (-2t+2)³/2 (减速段)
  // 效果：起步慢 → 中间快 → 结尾慢，模拟飞书的"飞入"效果
}
```

**动画期间的保护：**

- `isAnimatingRef.current = true` 阻止：
  - 滚动锚点恢复逻辑（避免恢复与动画冲突）
  - 无限滚动加载（避免动画经过顶部/底部时触发加载）
  - 滚动事件处理（避免动画中触发的 scroll 事件被误处理）

**高亮效果：**

目标消息在动画结束后设置 `highlightId`，对应的气泡会应用高亮样式：

```typescript
highlighted: {
  backgroundColor: '#fff3b0',                    // 淡黄色背景
  border: '2px solid #f5c542',                   // 金色边框
  boxShadow: '0 0 12px rgba(245, 197, 66, 0.4)', // 金色光晕
}
```

气泡本身有 `transition: 'background-color 0.5s ease, border-color 0.5s ease'`，所以高亮出现和消失都有平滑过渡。

#### 三、无限滚动加载

```typescript
const handleScroll = useCallback(() => {
  if (isAnimatingRef.current || scrollThrottleRef.current) return;
  if (!container || loading) return;

  // rAF 节流：一帧内只处理一次 scroll 事件
  scrollThrottleRef.current = true;
  requestAnimationFrame(() => { scrollThrottleRef.current = false; });

  // 距顶部 < 300px → 加载更早消息
  if (scrollTop < 300 && firstSegment?.hasMoreBefore) {
    anchorRef.current = getFirstVisibleAnchor();  // 加载前主动快照锚点
    onLoadBefore(firstSegment.id);
  }

  // 距底部 < 300px → 加载更新消息
  if (scrollHeight - scrollTop - clientHeight < 300 && lastSegment?.hasMoreAfter) {
    anchorRef.current = getFirstVisibleAnchor();
    onLoadAfter(lastSegment.id);
  }
}, [loading, firstSegment, lastSegment, ...]);
```

**节流策略：**

使用 `requestAnimationFrame` 节流而非 setTimeout/debounce，保证每一帧最多处理一次 scroll 事件，既不丢失触发时机，又不过度消耗性能。

**加载前主动快照：**

在调用 `onLoadBefore`/`onLoadAfter` 之前，主动执行 `anchorRef.current = getFirstVisibleAnchor()` 确保锚点数据是最新的，因为正常的快照（每次 paint 后的 useEffect）可能还没来得及执行。

#### 四、Gap 指示器

当存在多个未合并的 Segment 时，在相邻段之间渲染 gap 指示器：

```typescript
if (si > 0) {
  const prevLast = prevSeg.messages[prevSeg.messages.length - 1]?.id ?? 0;
  const currFirst = seg.messages[0]?.id ?? 0;
  if (currFirst > prevLast + 1) {
    // 渲染："此处有 {N} 条未加载消息"
    // 两侧有渐变分隔线
  }
}
```

Gap 数量通过 `currFirst - prevLast - 1` 精确计算。

#### 五、消息渲染

每条消息根据 `senderId` 决定布局方向：
- `senderId === 'me'`：`flexDirection: 'row-reverse'`，头像在右、气泡在右，使用蓝色背景（`#c6e4ff`）
- 其他发送者：`flexDirection: 'row'`，头像在左、气泡在左，使用白色背景（`#ffffff`），显示发送者名称

每条消息 DOM 元素上标记 `data-msg-id={msg.id}` 属性，供锚点定位和跳转查询使用。

---

## 侧边栏组件

### `Sidebar.tsx` — 跳转控制面板

- **自定义跳转**：输入框接受 1-2000 的消息编号，支持回车键触发
- **快捷跳转**：11 个预设按钮（#1 ~ #1950），覆盖消息列表不同区域
- **状态联动**：`loading` 为 true 时所有按钮置灰禁用，防止并发请求
- **功能说明**：底部展示 5 条使用提示

---

## 主布局

### `App.tsx` — 组件组装

布局结构：`flex` 水平排列
```
┌─────────────────────────────────────┬────────────┐
│           顶部栏 (60px 固定高度)      │            │
│  💬 飞书聊天消息 Demo                 │            │
│  已加载 N 条消息 · M 个消息段         │            │
├─────────────────────────────────────┤  侧边栏     │
│                                     │  (260px)    │
│           ChatList (flex:1)         │  跳转控制   │
│           消息列表                   │  快捷按钮   │
│           (可滚动区域)               │  功能说明   │
│                                     │            │
└─────────────────────────────────────┴────────────┘
```

顶部栏实时显示：
- 已加载消息总数（所有 Segment 的消息数之和）
- 当前消息段数量
- 加载状态 badge

---

## 全局样式

### `main.tsx` — 样式注入

通过 JavaScript 动态创建 `<style>` 标签注入全局样式：

- **CSS Reset**：`* { margin: 0; padding: 0; box-sizing: border-box; }`
- **滚动条美化**：6px 宽的细滚动条，透明轨道，灰色圆角滑块
- **旋转动画**：`@keyframes spin` 用于加载指示器的旋转效果
- **数字输入框**：隐藏 Chrome/Firefox 的原生数字上下箭头

---

## 完整数据流

### 场景一：首次进入会话

```
App mount
  → useEffect 调用 loadInitial()
    → fetchLatestMessages() 返回 #1971~#2000，300ms 延迟
    → 创建 Segment { id:'seg_1', messages:[#1971..#2000], hasMoreBefore:true, hasMoreAfter:false }
    → ChatList 收到 segments，allMessages.length 从 0 → 30
    → useLayoutEffect 检测首次渲染（wasNull=true），执行初始滚动：scrollTop = scrollHeight
    → 用户看到最新的 30 条消息，光标在底部
```

### 场景二：向上滚动加载更早消息

```
用户向上滚动
  → scrollTop < 300，触发 handleScroll
    → 主动快照锚点 anchorRef = { msgId: 1971, offset: 50 }
    → 调用 onLoadBefore('seg_1')
      → loadBefore 从 segmentsRef 取到 seg_1 的首条消息 id=1971
      → fetchMessagesBefore(1971) 返回 #1941~#1970，400ms 延迟
      → prepend 到 seg_1：messages 变为 [#1941..#2000]
      → mergeSegments 检查：只有一个段，无需合并
  → React 重渲染，DOM 更新，列表顶部新增了 30 条消息
  → useLayoutEffect 检测 currentFirstId 从 1971 → 1941
    → 找到锚点元素 [data-msg-id="1971"]
    → 计算新 scrollTop = 1971元素.offsetTop - 50
    → 直接设置 scrollTop，用户看到的内容完全不动
  → 用户无感知地获得了更早的消息
```

### 场景三：跳转到未加载的历史消息

```
用户在侧边栏输入 500，点击跳转
  → jumpToMessage(500)
    → segmentsRef 中检查：没有任何段包含 #500
    → fetchMessagesAround(500) 返回 #470~#530，500ms 延迟
    → 创建新段 { id:'seg_2', messages:[#470..#530], hasMoreBefore:true, hasMoreAfter:true }
    → mergeSegments([seg_1(#1941..#2000), seg_2(#470..#530)])
      → 排序：seg_2 在前（470 < 1941）
      → 检查 gap：530 + 1 = 531 < 1941，有 gap → 不合并
      → 结果：[seg_2, seg_1]，两段独立存在，中间有 gap 指示器
    → 设置 jumpingToId = 500

  → ChatList 收到 jumpingToId = 500
    → 找到 [data-msg-id="500"] 元素
    → 设置 isAnimatingRef = true（保护动画）
    → 计算目标 scrollTop：让 #500 出现在容器垂直中心
    → smoothScrollTo 600ms easeInOutCubic 动画
    → 动画结束：
      → isAnimatingRef = false
      → 高亮 #500 消息（黄色背景 + 金色边框 + 光晕）
      → 清除 jumpingToId
      → 重新快照锚点
      → 2秒后取消高亮
```

### 场景四：消息段合并

```
初始状态：
  seg_2: [#470..#530]  ← 跳转加载的历史消息段
  seg_1: [#1941..#2000] ← 首屏加载的最新消息段
  中间显示 gap："此处有 1410 条未加载消息"

用户在 seg_2 位置持续向下滚动：
  → loadAfter('seg_2') 每次加载 30 条
  → seg_2 逐步扩展：[#470..#560], [#470..#590], ..., [#470..#1940]

当 seg_2 扩展到 [#470..#1940]，再次 loadAfter：
  → fetchMessagesAfter(1940) 返回 #1941~#1970
  → seg_2 变为 [#470..#1970]
  → mergeSegments([seg_2(#470..#1970), seg_1(#1941..#2000)])
    → 排序：seg_2 在前
    → 检查：seg_2.最后=1970，seg_1.第一=1941，1941 ≤ 1970+1 → 重叠！
    → 合并：Map 去重 → [#470..#2000]
    → hasMoreBefore = seg_2.hasMoreBefore = true
    → hasMoreAfter = seg_1.hasMoreAfter = false
  → 结果：一个合并后的段 [#470..#2000]
  → gap 指示器消失，列表变成完全连续
  → 滚动锚点保持机制确保合并过程中视口位置不变
```

### 场景五：跳转到已加载的消息

```
当前已加载 [#470..#2000]
用户输入 800 跳转
  → jumpToMessage(800)
    → segmentsRef 检查：seg 包含 #800 → alreadyLoaded = true
    → 直接设置 jumpingToId = 800（不发起网络请求）
    → ChatList 收到跳转指令
    → 平滑滚动到 #800 位置 + 高亮
```

---

## 关键设计决策

### 1. 为什么用 Segment 模型而不是单一消息数组？

单一数组无法表达"中间有 gap"的状态。飞书等 IM 应用中，用户可能跳转到任意历史位置，产生多个不连续的已加载区域。Segment 模型天然支持这种离散加载场景，每个段独立维护边界状态。

### 2. 为什么合并条件是 `currFirst <= prevLast + 1`？

- `<=`：处理重叠场景（两次加载的区间有交集）
- `+1`：处理相邻场景（前一段最后是 #100，后一段第一条是 #101，逻辑上连续）

### 3. 为什么不用 `IntersectionObserver`？

`IntersectionObserver` 适合监测元素是否进入视口，但对于动态列表的滚动位置保持和精确锚点计算，直接读取 `scrollTop` + `offsetTop` 更可控。且 `useLayoutEffect` 中需要同步操作 DOM，`IntersectionObserver` 的异步回调不适用。

### 4. 为什么每次 paint 后都快照锚点？

```typescript
useEffect(() => {
  if (!isAnimatingRef.current) {
    anchorRef.current = getFirstVisibleAnchor();
  }
});  // 无依赖数组 → 每次渲染后都执行
```

因为无法预知下一次 prepend 何时发生。持续快照确保在任何时刻发生 prepend 时，都有一个最新的锚点可用。性能开销很小（只是遍历几个 DOM 元素读取 offsetTop）。

### 5. 为什么加载前还要主动快照一次？

```typescript
if (scrollTop < 300 && firstSegment?.hasMoreBefore) {
  anchorRef.current = getFirstVisibleAnchor();  // ← 这行
  onLoadBefore(firstSegment.id);
}
```

`useEffect` 的快照是在上一次 paint 后执行的，到 scroll 事件触发时用户可能已经继续滚动了。主动快照确保锚点反映的是"加载触发瞬间"的精确视口状态。
