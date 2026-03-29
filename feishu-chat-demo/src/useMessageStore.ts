import { useCallback, useRef, useState } from 'react';
import type { Segment } from './types';
import {
  fetchLatestMessages,
  fetchMessagesBefore,
  fetchMessagesAfter,
  fetchMessagesAround,
} from './mockData';

let segmentCounter = 0;
function nextSegmentId() {
  return `seg_${++segmentCounter}`;
}

function mergeSegments(segs: Segment[]): Segment[] {
  if (segs.length <= 1) return segs;

  const sorted = [...segs].sort((a, b) => {
    const aFirst = a.messages[0]?.id ?? 0;
    const bFirst = b.messages[0]?.id ?? 0;
    return aFirst - bFirst;
  });

  const merged: Segment[] = [{ ...sorted[0] }];

  for (let i = 1; i < sorted.length; i++) {
    const prev = merged[merged.length - 1];
    const curr = sorted[i];

    const prevLast = prev.messages[prev.messages.length - 1]?.id ?? 0;
    const currFirst = curr.messages[0]?.id ?? Infinity;

    if (currFirst <= prevLast + 1) {
      const msgMap = new Map(prev.messages.map(m => [m.id, m]));
      for (const m of curr.messages) msgMap.set(m.id, m);
      merged[merged.length - 1] = {
        id: prev.id,
        messages: Array.from(msgMap.values()).sort((a, b) => a.id - b.id),
        hasMoreBefore: prev.hasMoreBefore,
        hasMoreAfter: curr.hasMoreAfter,
      };
    } else {
      merged.push({ ...curr });
    }
  }

  return merged;
}

export function useMessageStore() {
  const [segments, setSegments] = useState<Segment[]>([]);
  const [loading, setLoading] = useState(false);
  const [jumpingToId, setJumpingToId] = useState<number | null>(null);
  const loadingRef = useRef(false);

  const loadInitial = useCallback(async () => {
    if (loadingRef.current) return;
    loadingRef.current = true;
    setLoading(true);
    try {
      const result = await fetchLatestMessages();
      const seg: Segment = {
        id: nextSegmentId(),
        messages: result.messages,
        hasMoreBefore: result.hasMoreBefore,
        hasMoreAfter: result.hasMoreAfter,
      };
      setSegments([seg]);
    } finally {
      setLoading(false);
      loadingRef.current = false;
    }
  }, []);

  const loadBefore = useCallback(async (segmentId: string) => {
    if (loadingRef.current) return;
    loadingRef.current = true;
    setLoading(true);
    try {
      let firstMsgId: number | null = null;
      setSegments(prev => {
        const seg = prev.find(s => s.id === segmentId);
        if (seg && seg.hasMoreBefore && seg.messages.length > 0) {
          firstMsgId = seg.messages[0].id;
        }
        return prev;
      });

      if (firstMsgId === null) {
        setLoading(false);
        loadingRef.current = false;
        return;
      }

      const result = await fetchMessagesBefore(firstMsgId);

      setSegments(prev => {
        if (result.messages.length === 0) {
          return prev.map(s =>
            s.id === segmentId ? { ...s, hasMoreBefore: false } : s
          );
        }
        const newSegs = prev.map(s => {
          if (s.id !== segmentId) return s;
          return {
            ...s,
            messages: [...result.messages, ...s.messages],
            hasMoreBefore: result.hasMoreBefore,
          };
        });
        return mergeSegments(newSegs);
      });
    } finally {
      setLoading(false);
      loadingRef.current = false;
    }
  }, []);

  const loadAfter = useCallback(async (segmentId: string) => {
    if (loadingRef.current) return;
    loadingRef.current = true;
    setLoading(true);
    try {
      let lastMsgId: number | null = null;
      setSegments(prev => {
        const seg = prev.find(s => s.id === segmentId);
        if (seg && seg.hasMoreAfter && seg.messages.length > 0) {
          lastMsgId = seg.messages[seg.messages.length - 1].id;
        }
        return prev;
      });

      if (lastMsgId === null) {
        setLoading(false);
        loadingRef.current = false;
        return;
      }

      const result = await fetchMessagesAfter(lastMsgId);

      setSegments(prev => {
        if (result.messages.length === 0) {
          return prev.map(s =>
            s.id === segmentId ? { ...s, hasMoreAfter: false } : s
          );
        }
        const newSegs = prev.map(s => {
          if (s.id !== segmentId) return s;
          return {
            ...s,
            messages: [...s.messages, ...result.messages],
            hasMoreAfter: result.hasMoreAfter,
          };
        });
        return mergeSegments(newSegs);
      });
    } finally {
      setLoading(false);
      loadingRef.current = false;
    }
  }, []);

  const jumpToMessage = useCallback(async (messageId: number) => {
    if (loadingRef.current) return;

    // Check if already loaded
    let found = false;
    setSegments(prev => {
      for (const seg of prev) {
        if (seg.messages.some(m => m.id === messageId)) {
          found = true;
          break;
        }
      }
      return prev;
    });

    if (found) {
      setJumpingToId(messageId);
      return;
    }

    loadingRef.current = true;
    setLoading(true);
    try {
      const result = await fetchMessagesAround(messageId);
      if (result.messages.length === 0) return;

      setSegments(prev => {
        const newSeg: Segment = {
          id: nextSegmentId(),
          messages: result.messages,
          hasMoreBefore: result.hasMoreBefore,
          hasMoreAfter: result.hasMoreAfter,
        };
        return mergeSegments([...prev, newSeg]);
      });
      setJumpingToId(messageId);
    } finally {
      setLoading(false);
      loadingRef.current = false;
    }
  }, []);

  const clearJumpTarget = useCallback(() => {
    setJumpingToId(null);
  }, []);

  return {
    segments,
    loading,
    jumpingToId,
    loadInitial,
    loadBefore,
    loadAfter,
    jumpToMessage,
    clearJumpTarget,
  };
}
