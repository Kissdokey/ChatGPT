export interface Message {
  id: number;
  content: string;
  senderId: string;
  senderName: string;
  avatar: string;
  timestamp: number;
}

export interface Segment {
  id: string;
  messages: Message[];
  hasMoreBefore: boolean;
  hasMoreAfter: boolean;
}

export interface FetchResult {
  messages: Message[];
  hasMoreBefore: boolean;
  hasMoreAfter: boolean;
}
