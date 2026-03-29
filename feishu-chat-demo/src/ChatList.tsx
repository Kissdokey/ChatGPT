import {
  useEffect,
  useLayoutEffect,
  useRef,
  useCallback,
  useState,
  type CSSProperties,
  type ReactNode,
} from 'react';
import type { Segment } from './types';

interface Props {
  segments: Segment[];
  loading: boolean;
  jumpingToId: number | null;
  onLoadBefore: (segmentId: string) => void;
  onLoadAfter: (segmentId: string) => void;
  onClearJump: () => void;
}

export default function ChatList({
  segments,
  loading,
  jumpingToId,
  onLoadBefore,
  onLoadAfter,
  onClearJump,
}: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [highlightId, setHighlightId] = useState<number | null>(null);
  const isAnimatingRef = useRef(false);

  // ===== Scroll Anchor Preservation =====
  // We track the first visible message and its visual offset before
  // any DOM change, and restore after layout so the user sees no jump.
  const anchorRef = useRef<{ msgId: number; offset: number } | null>(null);
  const prevFirstIdRef = useRef<number | null>(null);

  const allMessages = segments.flatMap(seg => seg.messages);
  const firstSegment = segments[0];
  const lastSegment = segments[segments.length - 1];

  const getFirstVisibleAnchor = useCallback(() => {
    const container = containerRef.current;
    if (!container) return null;
    const scrollTop = container.scrollTop;
    const children = container.querySelectorAll('[data-msg-id]');
    for (const child of children) {
      const el = child as HTMLElement;
      if (el.offsetTop + el.offsetHeight > scrollTop) {
        return {
          msgId: Number(el.dataset.msgId),
          offset: el.offsetTop - scrollTop,
        };
      }
    }
    return null;
  }, []);

  // Before DOM paints: snapshot anchor
  // We detect when messages change by comparing the first message id.
  const currentFirstId = allMessages[0]?.id ?? null;

  // Save anchor before the render that changes messages
  if (currentFirstId !== prevFirstIdRef.current && prevFirstIdRef.current !== null) {
    // Messages changed — we need the anchor from before this render.
    // anchorRef.current was already set during the previous render's commit phase.
  }

  // After every paint, snapshot the current anchor for the next render
  useEffect(() => {
    if (!isAnimatingRef.current) {
      anchorRef.current = getFirstVisibleAnchor();
    }
  });

  // When messages change (e.g. prepend), restore scroll position to maintain anchor
  useLayoutEffect(() => {
    if (currentFirstId === prevFirstIdRef.current) return;
    const wasNull = prevFirstIdRef.current === null;
    prevFirstIdRef.current = currentFirstId;

    if (wasNull || isAnimatingRef.current || jumpingToId !== null) return;

    const container = containerRef.current;
    const anchor = anchorRef.current;
    if (!container || !anchor) return;

    const anchorEl = container.querySelector(
      `[data-msg-id="${anchor.msgId}"]`
    ) as HTMLElement | null;
    if (!anchorEl) return;

    container.scrollTop = anchorEl.offsetTop - anchor.offset;
  }, [currentFirstId, jumpingToId]);

  // Also handle append case: track by last message id
  const prevLastIdRef = useRef<number | null>(null);
  const currentLastId = allMessages[allMessages.length - 1]?.id ?? null;

  useLayoutEffect(() => {
    if (currentLastId === prevLastIdRef.current) return;
    const wasNull = prevLastIdRef.current === null;
    prevLastIdRef.current = currentLastId;

    if (wasNull || isAnimatingRef.current || jumpingToId !== null) return;
    if (currentFirstId !== prevFirstIdRef.current) return; // prepend handled above

    const container = containerRef.current;
    const anchor = anchorRef.current;
    if (!container || !anchor) return;

    const anchorEl = container.querySelector(
      `[data-msg-id="${anchor.msgId}"]`
    ) as HTMLElement | null;
    if (!anchorEl) return;

    container.scrollTop = anchorEl.offsetTop - anchor.offset;
  }, [currentLastId, currentFirstId, jumpingToId]);

  // ===== Initial scroll to bottom =====
  const initialScrollDone = useRef(false);
  useLayoutEffect(() => {
    if (!initialScrollDone.current && allMessages.length > 0 && !jumpingToId) {
      const container = containerRef.current;
      if (container) {
        container.scrollTop = container.scrollHeight;
        initialScrollDone.current = true;
      }
    }
  }, [allMessages.length, jumpingToId]);

  // ===== Jump to message with smooth animation =====
  useEffect(() => {
    if (jumpingToId === null) return;
    const container = containerRef.current;
    if (!container) return;

    const targetEl = container.querySelector(
      `[data-msg-id="${jumpingToId}"]`
    ) as HTMLElement | null;
    if (!targetEl) return;

    isAnimatingRef.current = true;

    const containerHeight = container.clientHeight;
    const targetTop = targetEl.offsetTop;
    const targetCenter =
      targetTop - containerHeight / 2 + targetEl.offsetHeight / 2;

    smoothScrollTo(container, Math.max(0, targetCenter), 600).then(() => {
      isAnimatingRef.current = false;
      setHighlightId(jumpingToId);
      onClearJump();
      anchorRef.current = getFirstVisibleAnchor();
      setTimeout(() => setHighlightId(null), 2000);
    });
  }, [jumpingToId, allMessages.length, onClearJump, getFirstVisibleAnchor]);

  // ===== Infinite scroll =====
  const scrollThrottleRef = useRef(false);

  const handleScroll = useCallback(() => {
    if (isAnimatingRef.current || scrollThrottleRef.current) return;
    const container = containerRef.current;
    if (!container || loading) return;

    scrollThrottleRef.current = true;
    requestAnimationFrame(() => {
      scrollThrottleRef.current = false;
    });

    const { scrollTop, scrollHeight, clientHeight } = container;

    if (scrollTop < 300 && firstSegment?.hasMoreBefore) {
      anchorRef.current = getFirstVisibleAnchor();
      onLoadBefore(firstSegment.id);
    }

    if (
      scrollHeight - scrollTop - clientHeight < 300 &&
      lastSegment?.hasMoreAfter
    ) {
      anchorRef.current = getFirstVisibleAnchor();
      onLoadAfter(lastSegment.id);
    }
  }, [
    loading,
    firstSegment,
    lastSegment,
    onLoadBefore,
    onLoadAfter,
    getFirstVisibleAnchor,
  ]);

  // ===== Render =====
  const renderContent = (): ReactNode[] => {
    const items: ReactNode[] = [];

    for (let si = 0; si < segments.length; si++) {
      const seg = segments[si];

      if (si === 0 && seg.hasMoreBefore) {
        items.push(
          <div key={`top-loader-${seg.id}`} style={styles.loader}>
            {loading ? (
              <span style={styles.loadingSpinner}>⟳ 加载中...</span>
            ) : (
              '↑ 上滑加载更多'
            )}
          </div>
        );
      }

      // Gap indicator between disconnected segments
      if (si > 0) {
        const prevSeg = segments[si - 1];
        const prevLast =
          prevSeg.messages[prevSeg.messages.length - 1]?.id ?? 0;
        const currFirst = seg.messages[0]?.id ?? 0;
        if (currFirst > prevLast + 1) {
          items.push(
            <div key={`gap-${si}`} style={styles.gapIndicator}>
              <div style={styles.gapLine} />
              <span style={styles.gapText}>
                此处有 {currFirst - prevLast - 1} 条未加载消息
              </span>
              <div style={styles.gapLine} />
            </div>
          );
        }
      }

      for (const msg of seg.messages) {
        const isMe = msg.senderId === 'me';
        const isHighlighted = msg.id === highlightId;

        items.push(
          <div
            key={msg.id}
            data-msg-id={msg.id}
            style={{
              ...styles.messageRow,
              flexDirection: isMe ? 'row-reverse' : 'row',
            }}
          >
            <div style={styles.avatar}>{msg.avatar}</div>
            <div
              style={{
                ...styles.bubble,
                ...(isMe ? styles.bubbleMe : styles.bubbleOther),
                ...(isHighlighted ? styles.highlighted : {}),
              }}
            >
              {!isMe && (
                <div style={styles.senderName}>{msg.senderName}</div>
              )}
              <div style={styles.messageContent}>{msg.content}</div>
              <div style={styles.timestamp}>
                {new Date(msg.timestamp).toLocaleTimeString('zh-CN', {
                  hour: '2-digit',
                  minute: '2-digit',
                })}
              </div>
            </div>
          </div>
        );
      }

      if (si === segments.length - 1 && seg.hasMoreAfter) {
        items.push(
          <div key={`bottom-loader-${seg.id}`} style={styles.loader}>
            {loading ? (
              <span style={styles.loadingSpinner}>⟳ 加载中...</span>
            ) : (
              '↓ 下滑加载更多'
            )}
          </div>
        );
      }
    }

    return items;
  };

  return (
    <div ref={containerRef} style={styles.container} onScroll={handleScroll}>
      {allMessages.length === 0 && loading && (
        <div style={styles.centerLoader}>
          <div style={styles.loadingSpinner}>加载消息中...</div>
        </div>
      )}
      {renderContent()}
    </div>
  );
}

function smoothScrollTo(
  element: HTMLElement,
  target: number,
  duration: number
): Promise<void> {
  return new Promise(resolve => {
    const start = element.scrollTop;
    const distance = target - start;
    if (Math.abs(distance) < 1) {
      resolve();
      return;
    }
    let startTime: number | null = null;

    function easeInOutCubic(t: number): number {
      return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
    }

    function step(currentTime: number) {
      if (!startTime) startTime = currentTime;
      const elapsed = currentTime - startTime;
      const progress = Math.min(elapsed / duration, 1);

      element.scrollTop = start + distance * easeInOutCubic(progress);

      if (progress < 1) {
        requestAnimationFrame(step);
      } else {
        resolve();
      }
    }

    requestAnimationFrame(step);
  });
}

const styles: Record<string, CSSProperties> = {
  container: {
    flex: 1,
    overflowY: 'auto',
    overflowX: 'hidden',
    padding: '12px 16px',
    backgroundColor: '#f5f6f7',
  },
  loader: {
    textAlign: 'center',
    padding: '16px',
    color: '#8f959e',
    fontSize: '13px',
  },
  loadingSpinner: {
    display: 'inline-block',
    animation: 'spin 1s linear infinite',
    color: '#3370ff',
  },
  centerLoader: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    height: '100%',
    color: '#8f959e',
    fontSize: '14px',
  },
  gapIndicator: {
    display: 'flex',
    alignItems: 'center',
    gap: '12px',
    padding: '20px 0',
  },
  gapLine: {
    flex: 1,
    height: '1px',
    background:
      'linear-gradient(to right, transparent, #d0d3d6, transparent)',
  },
  gapText: {
    fontSize: '12px',
    color: '#8f959e',
    whiteSpace: 'nowrap',
  },
  messageRow: {
    display: 'flex',
    gap: '8px',
    marginBottom: '12px',
    alignItems: 'flex-start',
  },
  avatar: {
    width: '36px',
    height: '36px',
    borderRadius: '6px',
    backgroundColor: '#e8e9eb',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    fontSize: '20px',
    flexShrink: 0,
  },
  bubble: {
    maxWidth: '70%',
    padding: '8px 12px',
    borderRadius: '8px',
    fontSize: '14px',
    lineHeight: '1.6',
    position: 'relative' as const,
    transition: 'background-color 0.5s ease, border-color 0.5s ease',
  },
  bubbleOther: {
    backgroundColor: '#ffffff',
    border: '1px solid #e8e9eb',
  },
  bubbleMe: {
    backgroundColor: '#c6e4ff',
    border: '1px solid #a8d4f5',
  },
  highlighted: {
    backgroundColor: '#fff3b0',
    border: '2px solid #f5c542',
    boxShadow: '0 0 12px rgba(245, 197, 66, 0.4)',
  },
  senderName: {
    fontSize: '12px',
    color: '#8f959e',
    marginBottom: '2px',
  },
  messageContent: {
    color: '#1f2329',
    wordBreak: 'break-word' as const,
  },
  timestamp: {
    fontSize: '11px',
    color: '#8f959e',
    marginTop: '4px',
    textAlign: 'right' as const,
  },
};
