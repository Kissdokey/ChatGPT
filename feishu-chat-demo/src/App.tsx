import { useEffect, type CSSProperties } from 'react';
import ChatList from './ChatList';
import Sidebar from './Sidebar';
import { useMessageStore } from './useMessageStore';

export default function App() {
  const {
    segments,
    loading,
    jumpingToId,
    loadInitial,
    loadBefore,
    loadAfter,
    jumpToMessage,
    clearJumpTarget,
  } = useMessageStore();

  useEffect(() => {
    loadInitial();
  }, [loadInitial]);

  const totalLoaded = segments.reduce(
    (sum, seg) => sum + seg.messages.length,
    0
  );

  return (
    <div style={styles.root}>
      <div style={styles.chatContainer}>
        <div style={styles.header}>
          <div style={styles.headerLeft}>
            <div style={styles.groupAvatar}>💬</div>
            <div>
              <div style={styles.chatTitle}>飞书聊天消息 Demo</div>
              <div style={styles.chatSubtitle}>
                已加载 {totalLoaded} 条消息 · {segments.length} 个消息段
              </div>
            </div>
          </div>
          {loading && <div style={styles.loadingBadge}>加载中...</div>}
        </div>

        <ChatList
          segments={segments}
          loading={loading}
          jumpingToId={jumpingToId}
          onLoadBefore={loadBefore}
          onLoadAfter={loadAfter}
          onClearJump={clearJumpTarget}
        />
      </div>

      <Sidebar onJump={jumpToMessage} loading={loading} />
    </div>
  );
}

const styles: Record<string, CSSProperties> = {
  root: {
    display: 'flex',
    height: '100vh',
    width: '100vw',
    fontFamily:
      '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif',
    overflow: 'hidden',
  },
  chatContainer: {
    flex: 1,
    display: 'flex',
    flexDirection: 'column',
    minWidth: 0,
  },
  header: {
    height: '60px',
    backgroundColor: '#ffffff',
    borderBottom: '1px solid #e8e9eb',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: '0 20px',
    flexShrink: 0,
  },
  headerLeft: {
    display: 'flex',
    alignItems: 'center',
    gap: '12px',
  },
  groupAvatar: {
    width: '40px',
    height: '40px',
    borderRadius: '8px',
    backgroundColor: '#e8f0fe',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    fontSize: '22px',
  },
  chatTitle: {
    fontSize: '16px',
    fontWeight: 600,
    color: '#1f2329',
  },
  chatSubtitle: {
    fontSize: '12px',
    color: '#8f959e',
    marginTop: '2px',
  },
  loadingBadge: {
    padding: '4px 12px',
    backgroundColor: '#e8f0fe',
    borderRadius: '12px',
    fontSize: '12px',
    color: '#3370ff',
  },
};
