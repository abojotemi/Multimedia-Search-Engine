import streamlit as st
import time
from collections import deque
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

class PerformanceMonitor:
    def __init__(self, max_history=1000):
        self.max_history = max_history
        self.search_times = deque(maxlen=max_history)
        self.embedding_times = deque(maxlen=max_history)
        self.cache_hits = deque(maxlen=max_history)
        self.cache_misses = deque(maxlen=max_history)
        self.query_types = deque(maxlen=max_history)
        self.timestamps = deque(maxlen=max_history)
        
    def record_search(self, search_time, embedding_time, cache_hit, query_type):
        """Record metrics for a single search operation"""
        self.search_times.append(search_time)
        self.embedding_times.append(embedding_time)
        self.cache_hits.append(1 if cache_hit else 0)
        self.cache_misses.append(0 if cache_hit else 1)
        self.query_types.append(query_type)
        self.timestamps.append(datetime.now())
    
    def get_summary_stats(self):
        """Calculate summary statistics"""
        if not self.search_times:
            return None
        
        total_queries = len(self.search_times)
        cache_hit_rate = sum(self.cache_hits) / total_queries if total_queries > 0 else 0
        
        return {
            "total_queries": total_queries,
            "avg_search_time": np.mean(self.search_times),
            "avg_embedding_time": np.mean(self.embedding_times),
            "cache_hit_rate": cache_hit_rate,
            "text_query_count": sum(1 for qt in self.query_types if qt == "text"),
            "image_query_count": sum(1 for qt in self.query_types if qt == "image")
        }
    
    def create_response_time_plot(self):
        """Create response time trend plot"""
        df = pd.DataFrame({
            'timestamp': list(self.timestamps),
            'search_time': list(self.search_times),
            'embedding_time': list(self.embedding_times)
        })
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['search_time'],
                                mode='lines', name='Search Time'))
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['embedding_time'],
                                mode='lines', name='Embedding Time'))
        
        fig.update_layout(
            title='Response Time Trends',
            xaxis_title='Time',
            yaxis_title='Time (seconds)',
            height=400
        )
        return fig
    
    def create_cache_performance_plot(self):
        """Create cache performance plot"""
        df = pd.DataFrame({
            'timestamp': list(self.timestamps),
            'cache_hit': list(self.cache_hits)
        })
        
        # Calculate rolling average
        df['hit_rate'] = df['cache_hit'].rolling(window=20).mean()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['hit_rate'],
                                mode='lines', name='Cache Hit Rate'))
        
        fig.update_layout(
            title='Cache Performance',
            xaxis_title='Time',
            yaxis_title='Hit Rate',
            height=400
        )
        return fig
    
    def create_query_distribution_plot(self):
        """Create query type distribution plot"""
        query_counts = pd.Series(self.query_types).value_counts()
        
        fig = go.Figure(data=[go.Pie(labels=query_counts.index, 
                                    values=query_counts.values)])
        
        fig.update_layout(
            title='Query Type Distribution',
            height=400
        )
        return fig

def performance_metrics_interface():
    """Interface for displaying performance metrics"""
    st.header("Performance Metrics")
    
    # Initialize performance monitor in session state if not exists
    if 'performance_monitor' not in st.session_state:
        st.session_state.performance_monitor = PerformanceMonitor()
    
    monitor = st.session_state.performance_monitor
    stats = monitor.get_summary_stats()
    
    if not stats:
        st.info("No performance data available yet. Try performing some searches!")
        return
    
    # Display summary metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Queries", stats["total_queries"])
    with col2:
        st.metric("Avg Search Time", f"{stats['avg_search_time']:.3f}s")
    with col3:
        st.metric("Cache Hit Rate", f"{stats['cache_hit_rate']*100:.1f}%")
    
    # Display plots
    st.plotly_chart(monitor.create_response_time_plot(), use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(monitor.create_cache_performance_plot(), use_container_width=True)
    with col2:
        st.plotly_chart(monitor.create_query_distribution_plot(), use_container_width=True)
    
    # Additional statistics
    st.subheader("Detailed Statistics")
    st.write({
        "Text Queries": stats["text_query_count"],
        "Image Queries": stats["image_query_count"],
        "Average Embedding Time": f"{stats['avg_embedding_time']:.3f}s"
    })