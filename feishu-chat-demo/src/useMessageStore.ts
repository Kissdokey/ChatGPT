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
  const segmentsRef = useRef<Segment[]>([]);

  const updateSegments = useCallback((updater: (prev: Segment[]) => Segment[]) => {
    setSegments(prev => {
      const next = updater(prev);
      segmentsRef.current = next;
      return next;
    });
  }, []);

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
      updateSegments(() => [seg]);
    } finally {
      setLoading(false);
      loadingRef.current = false;
    }
  }, [updateSegments]);

  const loadBefore = useCallback(async (segmentId: string) => {
    if (loadingRef.current) return;
    loadingRef.current = true;
    setLoading(true);
    try {
      const seg = segmentsRef.current.find(s => s.id === segmentId);
      if (!seg || !seg.hasMoreBefore || seg.messages.length === 0) {
        setLoading(false);
        loadingRef.current = false;
        return;
      }

      const firstMsgId = seg.messages[0].id;
      const result = await fetchMessagesBefore(firstMsgId);

      updateSegments(prev => {
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
  }, [updateSegments]);

  const loadAfter = useCallback(async (segmentId: string) => {
    if (loadingRef.current) return;
    loadingRef.current = true;
    setLoading(true);
    try {
      const seg = segmentsRef.current.find(s => s.id === segmentId);
      if (!seg || !seg.hasMoreAfter || seg.messages.length === 0) {
        setLoading(false);
        loadingRef.current = false;
        return;
      }

      const lastMsgId = seg.messages[seg.messages.length - 1].id;
      const result = await fetchMessagesAfter(lastMsgId);

      updateSegments(prev => {
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
  }, [updateSegments]);

  const jumpToMessage = useCallback(async (messageId: number) => {
    if (loadingRef.current) return;

    const alreadyLoaded = segmentsRef.current.some(seg =>
      seg.messages.some(m => m.id === messageId)
    );

    if (alreadyLoaded) {
      setJumpingToId(messageId);
      return;
    }

    loadingRef.current = true;
    setLoading(true);
    try {
      const result = await fetchMessagesAround(messageId);
      if (result.messages.length === 0) return;

      updateSegments(prev => {
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
  }, [updateSegments]);

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
