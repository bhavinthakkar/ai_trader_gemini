import React, { useState, useEffect, useCallback } from 'react';
import {
  StyleSheet,
  Text,
  View,
  ScrollView,
  TouchableOpacity,
  TextInput,
  RefreshControl,
  ActivityIndicator,
  SafeAreaView,
  Modal,
  Alert,
} from 'react-native';
import { StatusBar } from 'expo-status-bar';

// Default backend API URL (Workstation Local Area Network IP)
const DEFAULT_API_URL = 'http://192.168.0.241:8000';

export default function App() {
  const [apiUrl, setApiUrl] = useState(DEFAULT_API_URL);
  const [activeTab, setActiveTab] = useState('signals'); // 'signals' | 'deepdive' | 'movers' | 'history' | 'settings'
  const [refreshing, setRefreshing] = useState(false);
  const [connected, setConnected] = useState(true);

  // Signals State
  const [latestSignals, setLatestSignals] = useState([]);
  const [summaryStats, setSummaryStats] = useState(null);
  const [signalDecisionFilter, setSignalDecisionFilter] = useState('ALL');
  const [signalSearch, setSignalSearch] = useState('');
  const [recencyDays, setRecencyDays] = useState(7);

  // Deep-Dive State
  const [selectedStock, setSelectedStock] = useState(null);
  const [inspectQuery, setInspectQuery] = useState('');
  const [analyzing, setAnalyzing] = useState(false);
  const [analysisStatus, setAnalysisStatus] = useState('');

  // Movers State
  const [activeMarket, setActiveMarket] = useState('US'); // 'US' | 'EU'
  const [activeMovers, setActiveMovers] = useState([]);
  const [loadingMovers, setLoadingMovers] = useState(false);

  // History State
  const [historyRecords, setHistoryRecords] = useState([]);
  const [historySearch, setHistorySearch] = useState('');

  // Settings State
  const [serverStatus, setServerStatus] = useState(null);
  const [evaluatingOutcomes, setEvaluatingOutcomes] = useState(false);

  // Fetch Signals & KPIs
  const loadSignalsData = useCallback(async () => {
    try {
      const summaryRes = await fetch(`${apiUrl}/api/signals/summary?days=${recencyDays}`);
      if (summaryRes.ok) {
        const sumData = await summaryRes.json();
        setSummaryStats(sumData);
        setConnected(true);
      }

      const signalsRes = await fetch(`${apiUrl}/api/signals/latest?days=${recencyDays}`);
      if (signalsRes.ok) {
        const sigData = await signalsRes.json();
        setLatestSignals(sigData.signals || []);
        if (sigData.signals && sigData.signals.length > 0 && !selectedStock) {
          setSelectedStock(sigData.signals[0]);
        }
      }
    } catch (err) {
      setConnected(false);
    }
  }, [apiUrl, recencyDays, selectedStock]);

  // Fetch Active Movers
  const loadMoversData = useCallback(async () => {
    setLoadingMovers(true);
    try {
      const res = await fetch(`${apiUrl}/api/stocks/active?market=${activeMarket}&limit=15`);
      if (res.ok) {
        const data = await res.json();
        setActiveMovers(data.stocks || []);
      }
    } catch (err) {
      setConnected(false);
    } finally {
      setLoadingMovers(false);
    }
  }, [apiUrl, activeMarket]);

  // Fetch History
  const loadHistoryData = useCallback(async () => {
    try {
      const res = await fetch(`${apiUrl}/api/signals/history?limit=100`);
      if (res.ok) {
        const data = await res.json();
        setHistoryRecords(data.records || []);
      }
    } catch (err) {
      setConnected(false);
    }
  }, [apiUrl]);

  // Initial load
  useEffect(() => {
    loadSignalsData();
  }, [loadSignalsData]);

  useEffect(() => {
    if (activeTab === 'movers') {
      loadMoversData();
    } else if (activeTab === 'history') {
      loadHistoryData();
    }
  }, [activeTab, loadMoversData, loadHistoryData]);

  const onRefresh = async () => {
    setRefreshing(true);
    if (activeTab === 'signals') await loadSignalsData();
    else if (activeTab === 'movers') await loadMoversData();
    else if (activeTab === 'history') await loadHistoryData();
    setRefreshing(false);
  };

  // Trigger Single Analysis
  const triggerAnalysis = async (symbol) => {
    if (!symbol || !symbol.trim()) return;
    setAnalyzing(true);
    setAnalysisStatus(`Queuing 6-agent analysis for ${symbol.trim().toUpperCase()}...`);
    try {
      const res = await fetch(`${apiUrl}/api/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          symbol: symbol.trim().toUpperCase(),
          model: 'bunny',
          is_eu: activeMarket === 'EU',
        }),
      });
      const data = await res.json();
      if (res.ok) {
        setAnalysisStatus(`Analysis queued (${data.job_id}). Waiting for agent synthesis...`);
        // Poll status
        const interval = setInterval(async () => {
          try {
            const jobRes = await fetch(`${apiUrl}/api/jobs/${data.job_id}`);
            const jobData = await jobRes.json();
            if (jobData.status === 'completed') {
              clearInterval(interval);
              setAnalyzing(false);
              setAnalysisStatus('');
              Alert.alert('✅ Analysis Complete', `Signal for ${symbol} has been generated and saved!`);
              loadSignalsData();
            } else if (jobData.status === 'failed' || jobData.status === 'error') {
              clearInterval(interval);
              setAnalyzing(false);
              setAnalysisStatus('');
              Alert.alert('⚠️ Analysis Failed', jobData.error || 'Execution encountered an error.');
            }
          } catch (e) {
            clearInterval(interval);
            setAnalyzing(false);
          }
        }, 3000);
      } else {
        setAnalyzing(false);
        Alert.alert('Error', data.detail || 'Failed to start analysis.');
      }
    } catch (err) {
      setAnalyzing(false);
      Alert.alert('Connection Error', `Could not reach server at ${apiUrl}`);
    }
  };

  // Filtered Signals
  const filteredSignals = latestSignals.filter((s) => {
    const matchesDecision = signalDecisionFilter === 'ALL' || s.decision === signalDecisionFilter;
    const matchesSearch = !signalSearch || s.symbol.toUpperCase().includes(signalSearch.toUpperCase());
    return matchesDecision && matchesSearch;
  });

  return (
    <SafeAreaView style={styles.safeArea}>
      <StatusBar style="light" />

      {/* Header */}
      <View style={styles.header}>
        <View>
          <Text style={styles.headerTitle}>📈 AI Trader</Text>
          <Text style={styles.headerSubtitle}>Autonomous Swing Trading Engine</Text>
        </View>
        <View style={[styles.statusPill, connected ? styles.statusConnected : styles.statusDisconnected]}>
          <Text style={styles.statusText}>{connected ? '● LIVE' : '○ OFFLINE'}</Text>
        </View>
      </View>

      {/* Main Tab Content */}
      <ScrollView
        style={styles.content}
        refreshControl={<RefreshControl refreshing={refreshing} onRefresh={onRefresh} tintColor="#3B82F6" />}
      >
        {/* ==================== TAB 1: SIGNALS ==================== */}
        {activeTab === 'signals' && (
          <View style={styles.tabContainer}>
            {/* KPI Cards Grid */}
            {summaryStats?.summary && (
              <View style={styles.kpiGrid}>
                <View style={styles.kpiCard}>
                  <Text style={styles.kpiLabel}>Tracked Stocks</Text>
                  <Text style={styles.kpiValue}>{summaryStats.summary.total_tracked}</Text>
                </View>
                <View style={styles.kpiCard}>
                  <Text style={styles.kpiLabel}>🟢 Buy Signals</Text>
                  <Text style={[styles.kpiValue, { color: '#10B981' }]}>{summaryStats.summary.buy_count}</Text>
                </View>
                <View style={styles.kpiCard}>
                  <Text style={styles.kpiLabel}>🔴 Sell Signals</Text>
                  <Text style={[styles.kpiValue, { color: '#EF4444' }]}>{summaryStats.summary.sell_count}</Text>
                </View>
                <View style={styles.kpiCard}>
                  <Text style={styles.kpiLabel}>⚪ Hold Signals</Text>
                  <Text style={[styles.kpiValue, { color: '#9CA3AF' }]}>{summaryStats.summary.hold_count}</Text>
                </View>
                <View style={styles.kpiCard}>
                  <Text style={styles.kpiLabel}>🎯 Win Rate</Text>
                  <Text style={[styles.kpiValue, { color: '#38BDF8' }]}>
                    {summaryStats.performance?.win_rate_pct ? `${summaryStats.performance.win_rate_pct}%` : 'Pending'}
                  </Text>
                </View>
                <View style={styles.kpiCard}>
                  <Text style={styles.kpiLabel}>📅 Window</Text>
                  <Text style={styles.kpiValue}>{recencyDays ? `${recencyDays}d` : 'All'}</Text>
                </View>
              </View>
            )}

            {/* Recency Selector Pills */}
            <View style={styles.recencyRow}>
              {[
                { label: '7 Days', val: 7 },
                { label: '14 Days', val: 14 },
                { label: '30 Days', val: 30 },
                { label: 'All Time', val: null },
              ].map((item) => (
                <TouchableOpacity
                  key={item.label}
                  style={[styles.recencyPill, recencyDays === item.val && styles.recencyPillActive]}
                  onPress={() => setRecencyDays(item.val)}
                >
                  <Text style={[styles.recencyPillText, recencyDays === item.val && styles.recencyPillTextActive]}>
                    {item.label}
                  </Text>
                </TouchableOpacity>
              ))}
            </View>

            {/* Search & Decision Filter */}
            <View style={styles.filterSection}>
              <TextInput
                style={styles.searchInput}
                placeholder="Search symbol (e.g. NVDA, AAPL)..."
                placeholderTextColor="#64748B"
                value={signalSearch}
                onChangeText={setSignalSearch}
              />
              <View style={styles.decisionFilterRow}>
                {['ALL', 'BUY', 'SELL', 'HOLD'].map((dec) => (
                  <TouchableOpacity
                    key={dec}
                    style={[styles.filterChip, signalDecisionFilter === dec && styles.filterChipActive]}
                    onPress={() => setSignalDecisionFilter(dec)}
                  >
                    <Text
                      style={[styles.filterChipText, signalDecisionFilter === dec && styles.filterChipTextActive]}
                    >
                      {dec}
                    </Text>
                  </TouchableOpacity>
                ))}
              </View>
            </View>

            {/* Signal List Cards */}
            <Text style={styles.sectionHeading}>
              Active Recommendations ({filteredSignals.length})
            </Text>

            {filteredSignals.length === 0 ? (
              <View style={styles.emptyCard}>
                <Text style={styles.emptyText}>No signals matching current filters.</Text>
              </View>
            ) : (
              filteredSignals.map((item) => {
                const isBuy = item.decision === 'BUY';
                const isSell = item.decision === 'SELL';
                const badgeColor = isBuy ? '#059669' : isSell ? '#DC2626' : '#4B5563';
                const textColor = isBuy ? '#34D399' : isSell ? '#F87171' : '#D1D5DB';

                return (
                  <TouchableOpacity
                    key={item.id || item.symbol}
                    style={styles.signalCard}
                    onPress={() => {
                      setSelectedStock(item);
                      setActiveTab('deepdive');
                    }}
                  >
                    <View style={styles.signalHeader}>
                      <View>
                        <Text style={styles.signalSymbol}>{item.symbol}</Text>
                        <Text style={styles.signalModel}>{item.model_used || '6-Agent AI'}</Text>
                      </View>
                      <View style={[styles.badge, { backgroundColor: badgeColor }]}>
                        <Text style={[styles.badgeText, { color: textColor }]}>{item.decision}</Text>
                      </View>
                    </View>

                    <View style={styles.signalStatsRow}>
                      <View style={styles.statCol}>
                        <Text style={styles.statLabel}>Confidence</Text>
                        <Text style={styles.statVal}>
                          {item.confidence ? `${Math.round(item.confidence * 100)}%` : 'N/A'}
                        </Text>
                      </View>
                      <View style={styles.statCol}>
                        <Text style={styles.statLabel}>Quant Score</Text>
                        <Text style={styles.statVal}>{item.quant_score ? `${item.quant_score}/100` : 'N/A'}</Text>
                      </View>
                      <View style={styles.statCol}>
                        <Text style={styles.statLabel}>Entry Price</Text>
                        <Text style={styles.statVal}>{item.entry_price ? `$${item.entry_price.toFixed(2)}` : 'N/A'}</Text>
                      </View>
                      <View style={styles.statCol}>
                        <Text style={styles.statLabel}>RVOL</Text>
                        <Text style={styles.statVal}>{item.rvol_20d ? `${item.rvol_20d.toFixed(1)}x` : 'N/A'}</Text>
                      </View>
                    </View>

                    {item.reason ? (
                      <Text style={styles.signalReason} numberOfLines={2}>
                        {item.reason}
                      </Text>
                    ) : null}

                    <View style={styles.cardFooter}>
                      <Text style={styles.cardTimestamp}>🕒 {item.timestamp}</Text>
                      <Text style={styles.cardLink}>Inspect Details →</Text>
                    </View>
                  </TouchableOpacity>
                );
              })
            )}
          </View>
        )}

        {/* ==================== TAB 2: DEEP-DIVE ==================== */}
        {activeTab === 'deepdive' && (
          <View style={styles.tabContainer}>
            {/* Quick ISIN / Ticker Analyzer Bar */}
            <View style={styles.lookupBox}>
              <Text style={styles.lookupTitle}>🔎 ISIN / European Ticker Synthesis</Text>
              <View style={styles.lookupInputRow}>
                <TextInput
                  style={[styles.searchInput, { flex: 1, marginBottom: 0 }]}
                  placeholder="ISIN, WKN, or Symbol (e.g. NVDA, US67066G1040)..."
                  placeholderTextColor="#64748B"
                  value={inspectQuery}
                  onChangeText={setInspectQuery}
                  autoCapitalize="characters"
                />
                <TouchableOpacity
                  style={[styles.actionBtn, analyzing && styles.actionBtnDisabled]}
                  disabled={analyzing}
                  onPress={() => triggerAnalysis(inspectQuery)}
                >
                  <Text style={styles.actionBtnText}>{analyzing ? 'Running...' : '🚀 Analyze'}</Text>
                </TouchableOpacity>
              </View>
              {analysisStatus ? (
                <View style={styles.statusBox}>
                  <ActivityIndicator size="small" color="#3B82F6" />
                  <Text style={styles.statusBoxText}>{analysisStatus}</Text>
                </View>
              ) : null}
            </View>

            {/* Selected Stock Inspection */}
            {selectedStock ? (
              <View style={styles.detailContainer}>
                <View style={styles.detailHeader}>
                  <View>
                    <Text style={styles.detailTitle}>{selectedStock.symbol}</Text>
                    <Text style={styles.detailSubtitle}>Model: {selectedStock.model_used || '6-Agent Ensemble'}</Text>
                  </View>
                  <View
                    style={[
                      styles.badge,
                      {
                        backgroundColor:
                          selectedStock.decision === 'BUY'
                            ? '#059669'
                            : selectedStock.decision === 'SELL'
                            ? '#DC2626'
                            : '#4B5563',
                      },
                    ]}
                  >
                    <Text style={styles.badgeText}>{selectedStock.decision}</Text>
                  </View>
                </View>

                {/* 5-Pillar Score Cards */}
                <Text style={styles.subHeading}>🧮 5-Pillar Quantitative Scores</Text>
                <View style={styles.pillarGrid}>
                  <View style={styles.pillarCard}>
                    <Text style={styles.pillarLabel}>Trend (25%)</Text>
                    <Text style={styles.pillarVal}>{selectedStock.trend_score ?? 'N/A'}</Text>
                  </View>
                  <View style={styles.pillarCard}>
                    <Text style={styles.pillarLabel}>Sector Rel (20%)</Text>
                    <Text style={styles.pillarVal}>{selectedStock.sector_score ?? 'N/A'}</Text>
                  </View>
                  <View style={styles.pillarCard}>
                    <Text style={styles.pillarLabel}>Alpha (20%)</Text>
                    <Text style={styles.pillarVal}>{selectedStock.alpha_score ?? 'N/A'}</Text>
                  </View>
                  <View style={styles.pillarCard}>
                    <Text style={styles.pillarLabel}>Val Hist (15%)</Text>
                    <Text style={styles.pillarVal}>{selectedStock.val_history_score ?? 'N/A'}</Text>
                  </View>
                  <View style={styles.pillarCard}>
                    <Text style={styles.pillarLabel}>Peer Val (20%)</Text>
                    <Text style={styles.pillarVal}>{selectedStock.peer_val_score ?? 'N/A'}</Text>
                  </View>
                </View>

                {/* Stops, Targets & Setup Geometry */}
                <Text style={styles.subHeading}>⚖️ Execution Geometry & Risk Profile</Text>
                <View style={styles.geometryGrid}>
                  <View style={styles.geoCard}>
                    <Text style={styles.geoLabel}>Stop Loss</Text>
                    <Text style={[styles.geoVal, { color: '#EF4444' }]}>
                      {selectedStock.stop_loss_price ? `$${selectedStock.stop_loss_price.toFixed(2)}` : 'N/A'}
                    </Text>
                  </View>
                  <View style={styles.geoCard}>
                    <Text style={styles.geoLabel}>Target Price</Text>
                    <Text style={[styles.geoVal, { color: '#10B981' }]}>
                      {selectedStock.target_price ? `$${selectedStock.target_price.toFixed(2)}` : 'N/A'}
                    </Text>
                  </View>
                  <View style={styles.geoCard}>
                    <Text style={styles.geoLabel}>Reward:Risk</Text>
                    <Text style={styles.geoVal}>{selectedStock.reward_risk_ratio || 'N/A'}</Text>
                  </View>
                  <View style={styles.geoCard}>
                    <Text style={styles.geoLabel}>RSI(14)</Text>
                    <Text style={styles.geoVal}>
                      {selectedStock.rsi14 ? selectedStock.rsi14.toFixed(1) : 'N/A'}
                    </Text>
                  </View>
                </View>

                {/* Bull Case Narrative */}
                <View style={styles.narrativeBox}>
                  <Text style={styles.narrativeTitle}>💡 Bull Case Rationale</Text>
                  <Text style={styles.narrativeText}>
                    {selectedStock.bull_case || selectedStock.reason || 'No detailed rationale.'}
                  </Text>
                </View>

                {/* Bear Case Risks */}
                <View style={[styles.narrativeBox, { borderColor: '#7F1D1D' }]}>
                  <Text style={[styles.narrativeTitle, { color: '#F87171' }]}>⚠️ Downside Risks & Bear Case</Text>
                  <Text style={styles.narrativeText}>
                    {selectedStock.bear_case || selectedStock.risk_assessment || 'Low risk setup.'}
                  </Text>
                </View>

                {/* SEC Filings & Macro */}
                {selectedStock.institutional_data ? (
                  <View style={styles.narrativeBox}>
                    <Text style={styles.narrativeTitle}>🏛️ SEC Institutional & Filings</Text>
                    <Text style={styles.narrativeText}>{selectedStock.institutional_data}</Text>
                  </View>
                ) : null}
              </View>
            ) : (
              <View style={styles.emptyCard}>
                <Text style={styles.emptyText}>Select a stock from the Signals tab or search above.</Text>
              </View>
            )}
          </View>
        )}

        {/* ==================== TAB 3: MOVERS ==================== */}
        {activeTab === 'movers' && (
          <View style={styles.tabContainer}>
            {/* Market Toggle */}
            <View style={styles.marketToggleRow}>
              <TouchableOpacity
                style={[styles.marketBtn, activeMarket === 'US' && styles.marketBtnActive]}
                onPress={() => setActiveMarket('US')}
              >
                <Text style={[styles.marketBtnText, activeMarket === 'US' && styles.marketBtnTextActive]}>
                  🇺🇸 US Markets (NYSE/NASDAQ)
                </Text>
              </TouchableOpacity>
              <TouchableOpacity
                style={[styles.marketBtn, activeMarket === 'EU' && styles.marketBtnActive]}
                onPress={() => setActiveMarket('EU')}
              >
                <Text style={[styles.marketBtnText, activeMarket === 'EU' && styles.marketBtnTextActive]}>
                  🇪🇺 Europe (gettex/XETRA)
                </Text>
              </TouchableOpacity>
            </View>

            <Text style={styles.sectionHeading}>
              🔥 High-Volume Movers [{activeMarket}] ({activeMovers.length})
            </Text>

            {loadingMovers ? (
              <View style={styles.loadingBox}>
                <ActivityIndicator size="large" color="#3B82F6" />
                <Text style={styles.loadingText}>Fetching active volume screener...</Text>
              </View>
            ) : activeMovers.length === 0 ? (
              <View style={styles.emptyCard}>
                <Text style={styles.emptyText}>No active market movers found.</Text>
              </View>
            ) : (
              activeMovers.map((m) => {
                const isPositive = (m.change_pct || 0) >= 0;
                return (
                  <View key={m.symbol} style={styles.moverCard}>
                    <View style={styles.moverHeader}>
                      <View>
                        <Text style={styles.moverSymbol}>{m.symbol}</Text>
                        <Text style={styles.moverName} numberOfLines={1}>{m.name}</Text>
                      </View>
                      <View style={{ alignItems: 'flex-end' }}>
                        <Text style={styles.moverPrice}>
                          {m.currency_symbol || '$'}{m.price?.toFixed(2)}
                        </Text>
                        <Text style={[styles.moverChange, { color: isPositive ? '#10B981' : '#EF4444' }]}>
                          {isPositive ? '+' : ''}{m.change_pct?.toFixed(2)}%
                        </Text>
                      </View>
                    </View>

                    <View style={styles.moverStats}>
                      <Text style={styles.moverStatText}>Vol: {(m.volume / 1e6).toFixed(1)}M</Text>
                      <Text style={styles.moverStatText}>RVOL: {m.rvol ? `${m.rvol.toFixed(1)}x` : 'N/A'}</Text>
                      <Text style={styles.moverStatText}>Catalyst: {m.catalyst_type || 'N/A'}</Text>
                    </View>

                    {m.reason_summary ? (
                      <Text style={styles.moverReason} numberOfLines={2}>
                        {m.reason_summary}
                      </Text>
                    ) : null}

                    <TouchableOpacity
                      style={styles.moverActionBtn}
                      onPress={() => {
                        setInspectQuery(m.symbol);
                        setActiveTab('deepdive');
                        triggerAnalysis(m.symbol);
                      }}
                    >
                      <Text style={styles.moverActionBtnText}>⚡ Analyze with bunny</Text>
                    </TouchableOpacity>
                  </View>
                );
              })
            )}
          </View>
        )}

        {/* ==================== TAB 4: HISTORY ==================== */}
        {activeTab === 'history' && (
          <View style={styles.tabContainer}>
            <TextInput
              style={styles.searchInput}
              placeholder="Search historical symbol..."
              placeholderTextColor="#64748B"
              value={historySearch}
              onChangeText={setHistorySearch}
            />
            <Text style={styles.sectionHeading}>Past Signals Log ({historyRecords.length})</Text>

            {historyRecords
              .filter((r) => !historySearch || r.symbol.toUpperCase().includes(historySearch.toUpperCase()))
              .map((rec) => (
                <View key={rec.id} style={styles.historyCard}>
                  <View style={styles.signalHeader}>
                    <Text style={styles.signalSymbol}>{rec.symbol}</Text>
                    <Text style={styles.historyDec}>{rec.decision}</Text>
                  </View>
                  <Text style={styles.historyReason} numberOfLines={2}>{rec.reason}</Text>
                  <Text style={styles.cardTimestamp}>🕒 {rec.timestamp} | {rec.model_used}</Text>
                </View>
              ))}
          </View>
        )}

        {/* ==================== TAB 5: SETTINGS ==================== */}
        {activeTab === 'settings' && (
          <View style={styles.tabContainer}>
            <Text style={styles.sectionHeading}>⚙️ Server & Network Configuration</Text>
            <View style={styles.settingsCard}>
              <Text style={styles.settingsLabel}>Backend API Base URL</Text>
              <TextInput
                style={styles.searchInput}
                value={apiUrl}
                onChangeText={setApiUrl}
                placeholder="http://192.168.0.x:8000"
                placeholderTextColor="#64748B"
                autoCapitalize="none"
              />
              <TouchableOpacity
                style={styles.actionBtn}
                onPress={async () => {
                  try {
                    const res = await fetch(`${apiUrl}/api/health`);
                    const data = await res.json();
                    if (res.ok) {
                      setConnected(true);
                      Alert.alert('✅ Connected!', `API reachable at ${apiUrl} (${data.version})`);
                    }
                  } catch (e) {
                    setConnected(false);
                    Alert.alert('❌ Connection Failed', `Could not reach ${apiUrl}`);
                  }
                }}
              >
                <Text style={styles.actionBtnText}>Test Server Connection</Text>
              </TouchableOpacity>
            </View>

            <Text style={styles.sectionHeading}>🎯 Model Performance & Forward Evaluation</Text>
            <View style={styles.settingsCard}>
              <Text style={styles.settingsDesc}>
                Compares historical recommendations against forward market bars to compute win rate.
              </Text>
              <TouchableOpacity
                style={[styles.actionBtn, evaluatingOutcomes && styles.actionBtnDisabled]}
                disabled={evaluatingOutcomes}
                onPress={async () => {
                  setEvaluatingOutcomes(true);
                  try {
                    const res = await fetch(`${apiUrl}/api/outcomes/evaluate`, { method: 'POST' });
                    const data = await res.json();
                    Alert.alert('Evaluation Complete', `Evaluated ${data.evaluated_count} trade outcomes.`);
                    loadSignalsData();
                  } catch (e) {
                    Alert.alert('Error', 'Failed to evaluate outcomes.');
                  } finally {
                    setEvaluatingOutcomes(false);
                  }
                }}
              >
                <Text style={styles.actionBtnText}>
                  {evaluatingOutcomes ? 'Evaluating...' : '⚡ Evaluate Forward Outcomes'}
                </Text>
              </TouchableOpacity>
            </View>

            <Text style={styles.sectionHeading}>🗑️ Database Maintenance</Text>
            <View style={styles.settingsCard}>
              <Text style={styles.settingsDesc}>
                Permanently purges signals older than 7 days from the SQLite database.
              </Text>
              <TouchableOpacity
                style={[styles.actionBtn, { backgroundColor: '#7F1D1D' }]}
                onPress={() => {
                  Alert.alert('Confirm Purge', 'Permanently delete signals older than 7 days?', [
                    { text: 'Cancel', style: 'cancel' },
                    {
                      text: 'Delete',
                      style: 'destructive',
                      onPress: async () => {
                        try {
                          const res = await fetch(`${apiUrl}/api/signals/purge`, {
                            method: 'DELETE',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ days: 7 }),
                          });
                          const data = await res.json();
                          Alert.alert('Purged', data.message);
                          loadSignalsData();
                        } catch (e) {
                          Alert.alert('Error', 'Failed to purge signals.');
                        }
                      },
                    },
                  ]);
                }}
              >
                <Text style={styles.actionBtnText}>Permanently Purge Old Signals (&gt; 7 Days)</Text>
              </TouchableOpacity>
            </View>
          </View>
        )}
      </ScrollView>

      {/* Bottom Navigation Bar */}
      <View style={styles.bottomNav}>
        {[
          { key: 'signals', label: 'Signals', icon: '📊' },
          { key: 'deepdive', label: 'Deep-Dive', icon: '🔍' },
          { key: 'movers', label: 'Movers', icon: '🔥' },
          { key: 'history', label: 'History', icon: '📜' },
          { key: 'settings', label: 'Settings', icon: '⚙️' },
        ].map((tab) => (
          <TouchableOpacity
            key={tab.key}
            style={[styles.navItem, activeTab === tab.key && styles.navItemActive]}
            onPress={() => setActiveTab(tab.key)}
          >
            <Text style={styles.navIcon}>{tab.icon}</Text>
            <Text style={[styles.navLabel, activeTab === tab.key && styles.navLabelActive]}>{tab.label}</Text>
          </TouchableOpacity>
        ))}
      </View>
    </SafeAreaView>
  );
}

// ==========================================
// Stylesheet (Dark Theme Finance UI)
// ==========================================

const styles = StyleSheet.create({
  safeArea: {
    flex: 1,
    backgroundColor: '#0B0E14',
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: 16,
    paddingTop: 45,
    paddingBottom: 12,
    backgroundColor: '#111827',
    borderBottomWidth: 1,
    borderBottomColor: '#1F2937',
  },
  headerTitle: {
    fontSize: 20,
    fontWeight: '800',
    color: '#F8FAFC',
    letterSpacing: 0.5,
  },
  headerSubtitle: {
    fontSize: 12,
    color: '#94A3B8',
  },
  statusPill: {
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 12,
  },
  statusConnected: {
    backgroundColor: '#064E3B',
    borderWidth: 1,
    borderColor: '#059669',
  },
  statusDisconnected: {
    backgroundColor: '#7F1D1D',
    borderWidth: 1,
    borderColor: '#DC2626',
  },
  statusText: {
    fontSize: 11,
    fontWeight: '700',
    color: '#F1F5F9',
  },
  content: {
    flex: 1,
    paddingHorizontal: 12,
  },
  tabContainer: {
    paddingVertical: 12,
    paddingBottom: 30,
  },
  kpiGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
    marginBottom: 12,
  },
  kpiCard: {
    flex: 1,
    minWidth: '30%',
    backgroundColor: '#151B26',
    borderRadius: 8,
    padding: 10,
    borderWidth: 1,
    borderColor: '#212C3D',
  },
  kpiLabel: {
    fontSize: 11,
    color: '#94A3B8',
    marginBottom: 4,
  },
  kpiValue: {
    fontSize: 16,
    fontWeight: '800',
    color: '#F8FAFC',
  },
  recencyRow: {
    flexDirection: 'row',
    gap: 6,
    marginBottom: 12,
  },
  recencyPill: {
    flex: 1,
    paddingVertical: 8,
    backgroundColor: '#1E293B',
    borderRadius: 6,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: '#334155',
  },
  recencyPillActive: {
    backgroundColor: '#2563EB',
    borderColor: '#3B82F6',
  },
  recencyPillText: {
    fontSize: 12,
    fontWeight: '600',
    color: '#94A3B8',
  },
  recencyPillTextActive: {
    color: '#FFFFFF',
  },
  filterSection: {
    marginBottom: 14,
  },
  searchInput: {
    backgroundColor: '#151B26',
    borderWidth: 1,
    borderColor: '#212C3D',
    borderRadius: 8,
    paddingHorizontal: 12,
    paddingVertical: 10,
    color: '#F1F5F9',
    fontSize: 14,
    marginBottom: 8,
  },
  decisionFilterRow: {
    flexDirection: 'row',
    gap: 6,
  },
  filterChip: {
    flex: 1,
    paddingVertical: 6,
    backgroundColor: '#151B26',
    borderRadius: 6,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: '#212C3D',
  },
  filterChipActive: {
    backgroundColor: '#1E293B',
    borderColor: '#60A5FA',
  },
  filterChipText: {
    fontSize: 12,
    fontWeight: '600',
    color: '#94A3B8',
  },
  filterChipTextActive: {
    color: '#60A5FA',
  },
  sectionHeading: {
    fontSize: 15,
    fontWeight: '700',
    color: '#E2E8F0',
    marginTop: 6,
    marginBottom: 10,
  },
  signalCard: {
    backgroundColor: '#151B26',
    borderRadius: 10,
    borderWidth: 1,
    borderColor: '#212C3D',
    padding: 12,
    marginBottom: 10,
  },
  signalHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 10,
  },
  signalSymbol: {
    fontSize: 18,
    fontWeight: '800',
    color: '#F8FAFC',
  },
  signalModel: {
    fontSize: 11,
    color: '#64748B',
  },
  badge: {
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 12,
  },
  badgeText: {
    fontSize: 12,
    fontWeight: '700',
  },
  signalStatsRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    backgroundColor: '#0E131F',
    borderRadius: 6,
    padding: 8,
    marginBottom: 8,
  },
  statCol: {
    alignItems: 'center',
  },
  statLabel: {
    fontSize: 10,
    color: '#64748B',
  },
  statVal: {
    fontSize: 13,
    fontWeight: '700',
    color: '#E2E8F0',
    marginTop: 2,
  },
  signalReason: {
    fontSize: 12,
    color: '#94A3B8',
    lineHeight: 18,
    marginBottom: 8,
  },
  cardFooter: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    borderTopWidth: 1,
    borderTopColor: '#1F2937',
    paddingTop: 8,
  },
  cardTimestamp: {
    fontSize: 11,
    color: '#64748B',
  },
  cardLink: {
    fontSize: 12,
    fontWeight: '600',
    color: '#38BDF8',
  },
  emptyCard: {
    backgroundColor: '#151B26',
    borderRadius: 8,
    padding: 24,
    alignItems: 'center',
  },
  emptyText: {
    color: '#64748B',
    fontSize: 14,
  },
  lookupBox: {
    backgroundColor: '#151B26',
    borderRadius: 10,
    borderWidth: 1,
    borderColor: '#212C3D',
    padding: 12,
    marginBottom: 14,
  },
  lookupTitle: {
    fontSize: 13,
    fontWeight: '700',
    color: '#E2E8F0',
    marginBottom: 8,
  },
  lookupInputRow: {
    flexDirection: 'row',
    gap: 8,
  },
  actionBtn: {
    backgroundColor: '#2563EB',
    paddingHorizontal: 14,
    borderRadius: 8,
    justifyContent: 'center',
    alignItems: 'center',
    minHeight: 44,
  },
  actionBtnDisabled: {
    backgroundColor: '#1E3A8A',
  },
  actionBtnText: {
    color: '#FFFFFF',
    fontWeight: '700',
    fontSize: 13,
  },
  statusBox: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    marginTop: 10,
    padding: 8,
    backgroundColor: '#0E131F',
    borderRadius: 6,
  },
  statusBoxText: {
    color: '#93C5FD',
    fontSize: 12,
  },
  detailContainer: {
    backgroundColor: '#151B26',
    borderRadius: 10,
    borderWidth: 1,
    borderColor: '#212C3D',
    padding: 14,
  },
  detailHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 14,
  },
  detailTitle: {
    fontSize: 22,
    fontWeight: '800',
    color: '#F8FAFC',
  },
  detailSubtitle: {
    fontSize: 12,
    color: '#64748B',
  },
  subHeading: {
    fontSize: 13,
    fontWeight: '700',
    color: '#CBD5E1',
    marginTop: 12,
    marginBottom: 8,
  },
  pillarGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 6,
  },
  pillarCard: {
    flex: 1,
    minWidth: '30%',
    backgroundColor: '#0E131F',
    borderRadius: 6,
    padding: 8,
    alignItems: 'center',
  },
  pillarLabel: {
    fontSize: 10,
    color: '#64748B',
  },
  pillarVal: {
    fontSize: 14,
    fontWeight: '700',
    color: '#38BDF8',
    marginTop: 2,
  },
  geometryGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 6,
  },
  geoCard: {
    flex: 1,
    minWidth: '45%',
    backgroundColor: '#0E131F',
    borderRadius: 6,
    padding: 8,
  },
  geoLabel: {
    fontSize: 10,
    color: '#64748B',
  },
  geoVal: {
    fontSize: 14,
    fontWeight: '700',
    color: '#F1F5F9',
    marginTop: 2,
  },
  narrativeBox: {
    backgroundColor: '#0E131F',
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#1E293B',
    padding: 10,
    marginTop: 10,
  },
  narrativeTitle: {
    fontSize: 12,
    fontWeight: '700',
    color: '#34D399',
    marginBottom: 4,
  },
  narrativeText: {
    fontSize: 12,
    color: '#94A3B8',
    lineHeight: 18,
  },
  marketToggleRow: {
    flexDirection: 'row',
    gap: 8,
    marginBottom: 12,
  },
  marketBtn: {
    flex: 1,
    paddingVertical: 10,
    backgroundColor: '#151B26',
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#212C3D',
    alignItems: 'center',
  },
  marketBtnActive: {
    backgroundColor: '#1E293B',
    borderColor: '#3B82F6',
  },
  marketBtnText: {
    fontSize: 12,
    fontWeight: '600',
    color: '#64748B',
  },
  marketBtnTextActive: {
    color: '#60A5FA',
    fontWeight: '700',
  },
  loadingBox: {
    padding: 40,
    alignItems: 'center',
  },
  loadingText: {
    color: '#64748B',
    fontSize: 13,
    marginTop: 10,
  },
  moverCard: {
    backgroundColor: '#151B26',
    borderRadius: 10,
    borderWidth: 1,
    borderColor: '#212C3D',
    padding: 12,
    marginBottom: 10,
  },
  moverHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 6,
  },
  moverSymbol: {
    fontSize: 17,
    fontWeight: '800',
    color: '#F8FAFC',
  },
  moverName: {
    fontSize: 12,
    color: '#64748B',
    maxWidth: 180,
  },
  moverPrice: {
    fontSize: 16,
    fontWeight: '800',
    color: '#F8FAFC',
  },
  moverChange: {
    fontSize: 13,
    fontWeight: '700',
  },
  moverStats: {
    flexDirection: 'row',
    gap: 12,
    marginVertical: 6,
  },
  moverStatText: {
    fontSize: 11,
    color: '#94A3B8',
  },
  moverReason: {
    fontSize: 12,
    color: '#64748B',
    marginBottom: 8,
  },
  moverActionBtn: {
    backgroundColor: '#1E293B',
    paddingVertical: 8,
    borderRadius: 6,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: '#334155',
  },
  moverActionBtnText: {
    color: '#38BDF8',
    fontSize: 12,
    fontWeight: '700',
  },
  historyCard: {
    backgroundColor: '#151B26',
    borderRadius: 8,
    padding: 10,
    marginBottom: 8,
    borderWidth: 1,
    borderColor: '#212C3D',
  },
  historyDec: {
    fontSize: 12,
    fontWeight: '700',
    color: '#94A3B8',
  },
  historyReason: {
    fontSize: 12,
    color: '#64748B',
    marginVertical: 4,
  },
  settingsCard: {
    backgroundColor: '#151B26',
    borderRadius: 10,
    borderWidth: 1,
    borderColor: '#212C3D',
    padding: 14,
    marginBottom: 14,
  },
  settingsLabel: {
    fontSize: 12,
    color: '#94A3B8',
    marginBottom: 6,
  },
  settingsDesc: {
    fontSize: 12,
    color: '#64748B',
    marginBottom: 10,
    lineHeight: 18,
  },
  bottomNav: {
    flexDirection: 'row',
    backgroundColor: '#111827',
    borderTopWidth: 1,
    borderTopColor: '#1F2937',
    paddingVertical: 8,
    paddingBottom: 22,
  },
  navItem: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
  },
  navItemActive: {},
  navIcon: {
    fontSize: 18,
    marginBottom: 3,
  },
  navLabel: {
    fontSize: 11,
    fontWeight: '600',
    color: '#64748B',
  },
  navLabelActive: {
    color: '#3B82F6',
    fontWeight: '700',
  },
});
