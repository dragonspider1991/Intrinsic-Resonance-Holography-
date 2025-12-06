/**
 * WebSocket Client for Real-time Updates
 * Handles WebSocket connections for job progress updates
 */

import type { JobResponse } from '../types';

const WS_BASE_URL = import.meta.env.VITE_WS_URL || 'ws://localhost:8000';

export class WebSocketClient {
  private ws: WebSocket | null = null;
  private jobId: string | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 1000;

  connect(jobId: string, onMessage: (data: JobResponse) => void, onError?: (error: Event) => void): void {
    this.jobId = jobId;
    const wsUrl = `${WS_BASE_URL}/ws/${jobId}`;

    try {
      this.ws = new WebSocket(wsUrl);

      this.ws.onopen = () => {
        console.log(`WebSocket connected for job ${jobId}`);
        this.reconnectAttempts = 0;
      };

      this.ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data) as JobResponse;
          onMessage(data);

          // Close connection if job is completed or failed
          if (data.status === 'completed' || data.status === 'failed') {
            this.disconnect();
          }
        } catch (error) {
          console.error('Error parsing WebSocket message:', error);
        }
      };

      this.ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        if (onError) {
          onError(error);
        }
      };

      this.ws.onclose = () => {
        console.log('WebSocket connection closed');
        
        // Attempt to reconnect if not manually disconnected
        if (this.jobId && this.reconnectAttempts < this.maxReconnectAttempts) {
          this.reconnectAttempts++;
          console.log(`Attempting to reconnect (${this.reconnectAttempts}/${this.maxReconnectAttempts})...`);
          
          setTimeout(() => {
            if (this.jobId) {
              this.connect(this.jobId, onMessage, onError);
            }
          }, this.reconnectDelay * this.reconnectAttempts);
        }
      };
    } catch (error) {
      console.error('Error creating WebSocket connection:', error);
      if (onError) {
        onError(error as Event);
      }
    }
  }

  disconnect(): void {
    if (this.ws) {
      this.jobId = null;
      this.reconnectAttempts = this.maxReconnectAttempts; // Prevent auto-reconnect
      this.ws.close();
      this.ws = null;
    }
  }

  isConnected(): boolean {
    return this.ws !== null && this.ws.readyState === WebSocket.OPEN;
  }
}

export const websocketClient = new WebSocketClient();
export default websocketClient;
