import { useState, type CSSProperties } from 'react';
import { getSearchableMessages } from './mockData';

interface Props {
  onJump: (messageId: number) => void;
  loading: boolean;
}

export default function Sidebar({ onJump, loading }: Props) {
  const [customId, setCustomId] = useState('');
  const bookmarks = getSearchableMessages();

  const handleJump = (id: number) => {
    if (!loading) {
      onJump(id);
    }
  };

  const handleCustomJump = () => {
    const id = parseInt(customId, 10);
    if (!isNaN(id) && id >= 1 && id <= 2000) {
      handleJump(id);
      setCustomId('');
    }
  };

  return (
    <div style={styles.sidebar}>
      <div style={styles.title}>跳转历史消息</div>
      <div style={styles.subtitle}>
        共 2000 条消息，输入 1-2000 的消息编号跳转
      </div>

      <div style={styles.inputRow}>
        <input
          style={styles.input}
          type="number"
          placeholder="输入消息编号"
          value={customId}
          onChange={e => setCustomId(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && handleCustomJump()}
          min={1}
          max={2000}
        />
        <button
          style={{
            ...styles.jumpBtn,
            ...(loading ? styles.jumpBtnDisabled : {}),
          }}
          onClick={handleCustomJump}
          disabled={loading}
        >
          跳转
        </button>
      </div>

      <div style={styles.divider} />

      <div style={styles.bookmarkTitle}>快捷跳转</div>
      <div style={styles.bookmarkList}>
        {bookmarks.map(b => (
          <button
            key={b.id}
            style={{
              ...styles.bookmarkBtn,
              ...(loading ? styles.bookmarkBtnDisabled : {}),
            }}
            onClick={() => handleJump(b.id)}
            disabled={loading}
          >
            #{b.id}
          </button>
        ))}
      </div>

      <div style={styles.divider} />

      <div style={styles.tips}>
        <div style={styles.tipTitle}>功能说明</div>
        <ul style={styles.tipList}>
          <li>点击跳转按钮，快速定位到历史消息</li>
          <li>跳转时加载目标消息前后各30条</li>
          <li>跳转后可以继续上下滚动加载</li>
          <li>当消息段与已加载消息重叠时自动合并</li>
          <li>滚动条会平滑过渡，不会跳动</li>
        </ul>
      </div>
    </div>
  );
}

const styles: Record<string, CSSProperties> = {
  sidebar: {
    width: '260px',
    backgroundColor: '#ffffff',
    borderLeft: '1px solid #e8e9eb',
    padding: '16px',
    overflowY: 'auto',
    display: 'flex',
    flexDirection: 'column',
    gap: '8px',
  },
  title: {
    fontSize: '16px',
    fontWeight: 600,
    color: '#1f2329',
  },
  subtitle: {
    fontSize: '12px',
    color: '#8f959e',
    marginBottom: '8px',
  },
  inputRow: {
    display: 'flex',
    gap: '8px',
  },
  input: {
    flex: 1,
    padding: '8px 12px',
    border: '1px solid #d0d3d6',
    borderRadius: '6px',
    fontSize: '13px',
    outline: 'none',
    minWidth: 0,
  },
  jumpBtn: {
    padding: '8px 16px',
    backgroundColor: '#3370ff',
    color: '#ffffff',
    border: 'none',
    borderRadius: '6px',
    fontSize: '13px',
    cursor: 'pointer',
    whiteSpace: 'nowrap',
  },
  jumpBtnDisabled: {
    backgroundColor: '#bfcfff',
    cursor: 'not-allowed',
  },
  divider: {
    height: '1px',
    backgroundColor: '#e8e9eb',
    margin: '8px 0',
  },
  bookmarkTitle: {
    fontSize: '13px',
    fontWeight: 500,
    color: '#646a73',
  },
  bookmarkList: {
    display: 'flex',
    flexWrap: 'wrap',
    gap: '6px',
  },
  bookmarkBtn: {
    padding: '4px 12px',
    backgroundColor: '#f0f1f2',
    border: '1px solid #e0e1e3',
    borderRadius: '14px',
    fontSize: '12px',
    color: '#3370ff',
    cursor: 'pointer',
    transition: 'all 0.2s',
  },
  bookmarkBtnDisabled: {
    color: '#bfcfff',
    cursor: 'not-allowed',
  },
  tips: {
    marginTop: '4px',
  },
  tipTitle: {
    fontSize: '13px',
    fontWeight: 500,
    color: '#646a73',
    marginBottom: '4px',
  },
  tipList: {
    fontSize: '12px',
    color: '#8f959e',
    paddingLeft: '16px',
    margin: 0,
    lineHeight: '1.8',
  },
};
