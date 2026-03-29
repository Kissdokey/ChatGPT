import type { Message, FetchResult } from './types';

const TOTAL_MESSAGES = 2000;
const PAGE_SIZE = 30;

const senders = [
  { id: 'user1', name: '张三', avatar: '🧑' },
  { id: 'user2', name: '李四', avatar: '👩' },
  { id: 'user3', name: '王五', avatar: '👨' },
  { id: 'user4', name: '赵六', avatar: '👱' },
  { id: 'me', name: '我', avatar: '😊' },
];

const contentTemplates = [
  '今天天气真不错，适合出去走走',
  '你看到那个新功能了吗？超好用的',
  '下午3点有个会议，别忘了',
  '这个Bug我已经修好了，你帮忙review一下',
  '周末有空一起吃饭吗？',
  '好的，收到',
  '👍',
  '这个方案我觉得可以，我们讨论一下细节',
  '刚才那个文档我已经更新了',
  '明天上午我有事，可能会晚点到',
  '这个需求的优先级是什么？',
  '我发了一个链接到群里，大家看看',
  '收到，马上处理',
  '这个版本什么时候发布？',
  '已经提交了MR，等待review',
  '辛苦了！',
  '好的，我来跟进这个事情',
  '需要我帮忙吗？',
  '那个设计稿已经出了，很好看',
  '这个问题我之前也遇到过',
];

const allMessages: Message[] = [];

for (let i = 1; i <= TOTAL_MESSAGES; i++) {
  const sender = senders[i % senders.length];
  const content = contentTemplates[i % contentTemplates.length];
  const baseTime = new Date('2026-01-01').getTime();
  allMessages.push({
    id: i,
    content: `[#${i}] ${content}`,
    senderId: sender.id,
    senderName: sender.name,
    avatar: sender.avatar,
    timestamp: baseTime + i * 60000,
  });
}

function delay(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

export async function fetchLatestMessages(): Promise<FetchResult> {
  await delay(300);
  const start = Math.max(0, TOTAL_MESSAGES - PAGE_SIZE);
  return {
    messages: allMessages.slice(start, TOTAL_MESSAGES),
    hasMoreBefore: start > 0,
    hasMoreAfter: false,
  };
}

export async function fetchMessagesBefore(messageId: number): Promise<FetchResult> {
  await delay(400);
  const idx = allMessages.findIndex(m => m.id === messageId);
  if (idx <= 0) {
    return { messages: [], hasMoreBefore: false, hasMoreAfter: true };
  }
  const start = Math.max(0, idx - PAGE_SIZE);
  return {
    messages: allMessages.slice(start, idx),
    hasMoreBefore: start > 0,
    hasMoreAfter: true,
  };
}

export async function fetchMessagesAfter(messageId: number): Promise<FetchResult> {
  await delay(400);
  const idx = allMessages.findIndex(m => m.id === messageId);
  if (idx < 0 || idx >= TOTAL_MESSAGES - 1) {
    return { messages: [], hasMoreBefore: true, hasMoreAfter: false };
  }
  const end = Math.min(TOTAL_MESSAGES, idx + 1 + PAGE_SIZE);
  return {
    messages: allMessages.slice(idx + 1, end),
    hasMoreBefore: true,
    hasMoreAfter: end < TOTAL_MESSAGES,
  };
}

export async function fetchMessagesAround(messageId: number): Promise<FetchResult> {
  await delay(500);
  const idx = allMessages.findIndex(m => m.id === messageId);
  if (idx < 0) {
    return { messages: [], hasMoreBefore: false, hasMoreAfter: false };
  }
  const start = Math.max(0, idx - PAGE_SIZE);
  const end = Math.min(TOTAL_MESSAGES, idx + PAGE_SIZE + 1);
  return {
    messages: allMessages.slice(start, end),
    hasMoreBefore: start > 0,
    hasMoreAfter: end < TOTAL_MESSAGES,
  };
}

export function getSearchableMessages(): { id: number; label: string }[] {
  const highlights = [1, 50, 100, 200, 500, 800, 1000, 1200, 1500, 1700, 1950];
  return highlights.map(id => ({
    id,
    label: `消息 #${id}`,
  }));
}
