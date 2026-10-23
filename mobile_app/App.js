import React, { useState, useEffect, useCallback, useMemo } from 'react';
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
  NativeModules,
} from 'react-native';
import { StatusBar } from 'expo-status-bar';

// Auto-detect backend API host from Metro bundler or fallback to Raspberry Pi
const getAutoApiUrl = () => {
  try {
    const scriptURL = NativeModules?.SourceCode?.scriptURL;
    if (scriptURL) {
      const match = scriptURL.match(/^https?:\/\/([^/:]+)/);
      if (match && match[1] && match[1] !== 'localhost' && match[1] !== '127.0.0.1') {
        return `http://${match[1]}:8000`;
      }
    }
  } catch (e) {}
  return 'http://192.168.0.92:8000';
};

// Default backend API URL
const DEFAULT_API_URL = getAutoApiUrl();

// Pre-curated Stock / Company Names Map (US & European Dual-Listings)
const STOCK_NAMES = {
  NVDA: 'NVIDIA Corporation',
  AAPL: 'Apple Inc.',
  TSLA: 'Tesla, Inc.',
  MSFT: 'Microsoft Corporation',
  AMZN: 'Amazon.com, Inc.',
  GOOGL: 'Alphabet Inc. (Class A)',
  GOOG: 'Alphabet Inc. (Class C)',
  GOOL: 'Alphabet Inc. (Class A)',
  META: 'Meta Platforms, Inc.',
  AMD: 'Advanced Micro Devices, Inc.',
  INTC: 'Intel Corporation',
  COIN: 'Coinbase Global, Inc.',
  AVGO: 'Broadcom Inc.',
  PLTR: 'Palantir Technologies Inc.',
  BABA: 'Alibaba Group Holding',
  BAC: 'Bank of America Corp.',
  NFLX: 'Netflix Inc.',
  SMCI: 'Super Micro Computer, Inc.',
  MSTR: 'MicroStrategy Inc.',
  QCOM: 'Qualcomm Inc.',
  ARM: 'Arm Holdings plc',
  MU: 'Micron Technology, Inc.',
  CRWD: 'CrowdStrike Holdings',
  PANW: 'Palo Alto Networks',
  CRM: 'Salesforce Inc.',
  ADBE: 'Adobe Inc.',
  ORCL: 'Oracle Corporation',
  CSCO: 'Cisco Systems Inc.',
  PYPL: 'PayPal Holdings, Inc.',
  SAP: 'SAP SE',
  SIE: 'Siemens AG',
  RHM: 'Rheinmetall AG',
  ALV: 'Allianz SE',
  DTE: 'Deutsche Telekom AG',
  MBG: 'Mercedes-Benz Group AG',
  BMW: 'Bayerische Motoren Werke AG',
  AIR: 'Airbus SE',
  IFX: 'Infineon Technologies AG',
  BAS: 'BASF SE',
  BAYN: 'Bayer AG',
  VOW3: 'Volkswagen AG',
  MUV2: 'Munich Re',
  DBK: 'Deutsche Bank AG',
  ASML: 'ASML Holding N.V.',
  MC: 'LVMH Moët Hennessy',
  JPM: 'JPMorgan Chase & Co.',
  V: 'Visa Inc.',
  DIS: 'The Walt Disney Company',
  FLKR: 'Flickr',
  // European dual-listings
  'NVD.DE': 'NVIDIA Corporation',
  'APC.DE': 'Apple Inc.',
  'TL0.DE': 'Tesla, Inc.',
  'MSF.DE': 'Microsoft Corporation',
  'AMZ.DE': 'Amazon.com, Inc.',
  'ABEA.DE': 'Alphabet Inc. (Class A)',
  'SET.DE': 'Meta Platforms, Inc.',
  'AMD.DE': 'Advanced Micro Devices, Inc.',
  '2PP.DE': 'PayPal Holdings, Inc.',
  'MQN.DE': 'Micron Technology, Inc.',
  'MQN.MU': 'Micron Technology, Inc.',
};

/**
 * Returns the human-readable stock/company name instead of the ticker symbol.
 */
function getStockName(itemOrSymbol) {
  if (!itemOrSymbol) return 'Unknown Stock';
  if (typeof itemOrSymbol === 'object') {
    const sym = (itemOrSymbol.symbol || itemOrSymbol.stock || itemOrSymbol.ticker || itemOrSymbol.query || '').trim().toUpperCase();
    const cleanSym = sym.split('.')[0];
    if (itemOrSymbol.company_name && itemOrSymbol.company_name.trim().toUpperCase() !== sym && itemOrSymbol.company_name.trim().toUpperCase() !== cleanSym) {
      return itemOrSymbol.company_name;
    }
    if (itemOrSymbol.name && itemOrSymbol.name.trim().toUpperCase() !== sym && itemOrSymbol.name.trim().toUpperCase() !== cleanSym) {
      return itemOrSymbol.name;
    }
    return STOCK_NAMES[sym] || STOCK_NAMES[cleanSym] || itemOrSymbol.company_name || itemOrSymbol.name || sym || 'Unknown Stock';
  }
  const sym = String(itemOrSymbol).trim().toUpperCase();
  const cleanSym = sym.split('.')[0];
  return STOCK_NAMES[sym] || STOCK_NAMES[cleanSym] || sym;
}

/**
 * Extracts and formats Standalone Point-in-Time ML Alpha indicators from a signal item or ML payload.
 */
function getMLInfo(item, supplementalML = null) {
  if (!item && !supplementalML) {
    return {
      available: false,
      score: null,
      scoreFormatted: 'N/A',
      label: 'Unavailable',
      badge: '⚪ N/A',
      tier: 'Unranked',
      stance: 'NEUTRAL',
      color: '#475569',
      bgColor: '#1E293B',
      textColor: '#94A3B8',
    };
  }

  // Parse raw_json if present
  let rawData = {};
  if (item && typeof item === 'object') {
    if (typeof item.raw_json === 'string') {
      try { rawData = JSON.parse(item.raw_json); } catch (e) {}
    } else if (typeof item.raw_json === 'object' && item.raw_json !== null) {
      rawData = item.raw_json;
    }
  }

  const mlSig = rawData.ml_signal || rawData.ml_pipeline_context || supplementalML?.data || supplementalML || null;
  const mlAssessment = rawData.ml_signal_assessment || null;
  const forecast20d = rawData.forecast_20d || null;

  // Derive score (0-100)
  let score = null;
  if (mlSig && mlSig.score_0_100 != null) {
    score = Number(mlSig.score_0_100);
  } else if (item && item.ml_score != null) {
    score = Number(item.ml_score);
  } else if (rawData.ml_score_0_100 != null) {
    score = Number(rawData.ml_score_0_100);
  } else if (supplementalML && supplementalML.score_0_100 != null) {
    score = Number(supplementalML.score_0_100);
  }

  if (score == null || isNaN(score)) {
    return {
      available: false,
      score: null,
      scoreFormatted: 'N/A',
      rank: null,
      universeSize: null,
      predictedReturn: null,
      predictedReturnFormatted: 'N/A',
      label: 'Non-S&P 500',
      badge: '⚪ Non-S&P 500',
      tier: 'Outside S&P 500 Universe',
      stance: 'NEUTRAL',
      color: '#475569',
      bgColor: '#151D2A',
      textColor: '#94A3B8',
      explanation: 'Constituent outside the S&P 500 coverage universe or pending ML score.',
    };
  }

  // Score tiers
  let label = 'MARKET PERFORM';
  let badge = '⚪ MARKET PERFORM';
  let tier = 'Quintile 3 (40-60% Alpha)';
  let stance = 'NEUTRAL';
  let color = '#94A3B8';
  let bgColor = '#1E293B';
  let textColor = '#F1F5F9';

  if (score >= 80) {
    label = 'STRONG OUTPERFORM';
    badge = '🟢 STRONG OUTPERFORM';
    tier = 'Top Quintile (Top 20% Alpha)';
    stance = 'BULLISH';
    color = '#10B981';
    bgColor = '#064E3B';
    textColor = '#34D399';
  } else if (score >= 60) {
    label = 'OUTPERFORM';
    badge = '🟢 OUTPERFORM';
    tier = 'Quintile 2 (60-80% Alpha)';
    stance = 'BULLISH';
    color = '#059669';
    bgColor = '#065F46';
    textColor = '#6EE7B7';
  } else if (score >= 40) {
    label = 'MARKET PERFORM';
    badge = '⚪ MARKET PERFORM';
    tier = 'Quintile 3 (40-60% Alpha)';
    stance = 'NEUTRAL';
    color = '#64748B';
    bgColor = '#1E293B';
    textColor = '#CBD5E1';
  } else if (score >= 20) {
    label = 'UNDERPERFORM';
    badge = '🔴 UNDERPERFORM';
    tier = 'Quintile 4 (20-40% Alpha)';
    stance = 'BEARISH';
    color = '#F59E0B';
    bgColor = '#451A03';
    textColor = '#FCD34D';
  } else {
    label = 'STRONG UNDERPERFORM';
    badge = '🔴 STRONG UNDERPERFORM';
    tier = 'Bottom Quintile (Bottom 20% Alpha)';
    stance = 'BEARISH';
    color = '#EF4444';
    bgColor = '#7F1D1D';
    textColor = '#FCA5A5';
  }

  // Rank and universe
  const rank = mlSig?.rank ?? item?.ml_rank ?? null;
  const universeSize = mlSig?.universe_size ?? item?.ml_universe_size ?? 483;

  // Predicted 20d return
  let predictedReturn = mlSig?.predicted_20d_excess_return ?? item?.ml_predicted_return ?? null;
  let predictedReturnFormatted = 'N/A';
  if (predictedReturn != null && !isNaN(Number(predictedReturn))) {
    const pct = Number(predictedReturn) * 100;
    predictedReturnFormatted = `${pct >= 0 ? '+' : ''}${pct.toFixed(2)}%`;
  }

  // Submodel ranks
  const rankFundamentals = mlSig?.rank_fundamentals ?? null;
  const rankTechnicals = mlSig?.rank_technical ?? null;
  const inTop10 = Boolean(mlSig?.in_top_10_portfolio || mlSig?.is_top10_pick);
  const scoreDate = mlSig?.score_date || null;
  const modelName = mlSig?.model_name || 'XGBoost 20d PIT Alpha';

  // LLM vs ML disposition
  const disposition = mlAssessment?.disposition || item?.ml_disposition || 'NEUTRAL';
  const dispositionReason = mlAssessment?.reason || null;
  const forecastOutlook = forecast20d?.relative_outlook || null;
  const forecastConfidence = forecast20d?.confidence != null ? Math.round(Number(forecast20d.confidence) * 100) : null;
  const forecastRationale = forecast20d?.rationale || null;

  return {
    available: true,
    score: Math.round(score * 10) / 10,
    scoreFormatted: `${(Math.round(score * 10) / 10).toFixed(1)}/100`,
    rank,
    universeSize,
    predictedReturn,
    predictedReturnFormatted,
    label: mlSig?.signal || item?.ml_signal_label || label,
    badge: mlSig?.badge || badge,
    tier: mlSig?.tier || tier,
    stance: mlSig?.stance || stance,
    color,
    bgColor,
    textColor,
    inTop10,
    rankFundamentals,
    rankTechnicals,
    scoreDate,
    modelName,
    disposition,
    dispositionReason,
    forecastOutlook,
    forecastConfidence,
    forecastRationale,
  };
}

/**
 * Extracts structured decision gates, programmatic reason codes, and position sizing (Steps 6.4 & 7).
 */
function getStructuredDecisionInfo(item) {
  if (!item) return null;
  let rawData = {};
  if (typeof item.raw_json === 'string') {
    try { rawData = JSON.parse(item.raw_json); } catch (e) {}
  } else if (typeof item.raw_json === 'object' && item.raw_json !== null) {
    rawData = item.raw_json;
  }

  const struct = item.structured_decision || rawData.structured_decision || {};
  const positionAction = item.position_action || struct.position_action || rawData.position_action || 'NO_ACTION';
  const passedGates = item.passed_gates || struct.passed_gates || rawData.passed_gates || [];
  const failedGates = item.failed_gates || struct.failed_gates || rawData.failed_gates || [];
  const reasonCodes = item.reason_codes || struct.reason_codes || rawData.reason_codes || [];
  const sizing = item.position_sizing || rawData.position_sizing || {};

  return {
    positionAction,
    passedGates: Array.isArray(passedGates) ? passedGates : [],
    failedGates: Array.isArray(failedGates) ? failedGates : [],
    reasonCodes: Array.isArray(reasonCodes) ? reasonCodes : [],
    sizing: typeof sizing === 'object' && sizing !== null ? sizing : {},
  };
}

export default function App() {
  const [apiUrl, setApiUrl] = useState(DEFAULT_API_URL);
  const [activeTab, setActiveTab] = useState('signals'); // 'signals' | 'deepdive' | 'shadow' | 'movers' | 'history' | 'settings'
  const [refreshing, setRefreshing] = useState(false);
  const [connected, setConnected] = useState(true);

  // Signals State
  const [latestSignals, setLatestSignals] = useState([]);
  const [summaryStats, setSummaryStats] = useState(null);
  const [signalDecisionFilter, setSignalDecisionFilter] = useState('ALL');
  const [signalSearch, setSignalSearch] = useState('');
  const [recencyDays, setRecencyDays] = useState(7);
  const [signalSortOrder, setSignalSortOrder] = useState('desc'); // 'desc' (newest analysis first) | 'asc' (oldest first)

  // Deep-Dive State
  const [selectedStock, setSelectedStock] = useState(null);
  const [selectedStockML, setSelectedStockML] = useState(null);
  const [loadingML, setLoadingML] = useState(false);
  const [inspectQuery, setInspectQuery] = useState('');
  const [analyzing, setAnalyzing] = useState(false);
  const [analysisStatus, setAnalysisStatus] = useState('');

  // Shadow Trading & Execution State (Steps 7.3 & 8)
  const [shadowDecisions, setShadowDecisions] = useState([]);
  const [shadowMetrics, setShadowMetrics] = useState(null);
  const [killSwitchState, setKillSwitchState] = useState(null);
  const [loadingShadow, setLoadingShadow] = useState(false);

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

      const signalsRes = await fetch(`${apiUrl}/api/signals/latest?days=${recencyDays}&sort_by=date&order=${signalSortOrder}`);
      if (signalsRes.ok) {
        const sigData = await signalsRes.json();
        const sigs = sigData.signals || [];
        setLatestSignals(sigs);
        if (sigs.length > 0) {
          if (!selectedStock || !sigs.some((s) => (s.symbol || s.stock) === (selectedStock.symbol || selectedStock.stock))) {
            setSelectedStock(sigs[0]);
          }
        } else if (!selectedStock) {
          // If no signals in current window, fetch all-time latest to ensure Deep-Dive has a stock populated
          try {
            const allRes = await fetch(`${apiUrl}/api/signals/latest?days=365&sort_by=date&order=desc`);
            if (allRes.ok) {
              const allData = await allRes.json();
              if (allData.signals && allData.signals.length > 0) {
                setSelectedStock(allData.signals[0]);
              }
            }
          } catch (e) {}
        }
      }
    } catch (err) {
      setConnected(false);
    }
  }, [apiUrl, recencyDays, selectedStock, signalSortOrder]);

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

  // Fetch Shadow Trading & Execution State (Steps 7.3 & 8)
  const loadShadowData = useCallback(async () => {
    setLoadingShadow(true);
    try {
      const [decRes, metRes, ksRes] = await Promise.all([
        fetch(`${apiUrl}/api/shadow/decisions?limit=25`),
        fetch(`${apiUrl}/api/shadow/metrics?limit=50`),
        fetch(`${apiUrl}/api/execution/kill-switch`),
      ]);
      if (decRes.ok) {
        const d = await decRes.json();
        setShadowDecisions(d.decisions || []);
      }
      if (metRes.ok) {
        const m = await metRes.json();
        setShadowMetrics(m);
      }
      if (ksRes.ok) {
        const ks = await ksRes.json();
        setKillSwitchState(ks);
      }
    } catch (err) {
      setConnected(false);
    } finally {
      setLoadingShadow(false);
    }
  }, [apiUrl]);

  // Operational Kill Switch Actions (Step 7.3)
  const triggerTripKillSwitch = async () => {
    try {
      const res = await fetch(`${apiUrl}/api/execution/kill-switch/trip`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ reason: 'MANUAL_OPERATOR_TRIP', operator: 'MobileApp' }),
      });
      if (res.ok) {
        await loadShadowData();
        Alert.alert('Kill Switch Tripped', 'All new risk entries have been frozen.');
      }
    } catch (err) {
      Alert.alert('Error', 'Failed to communicate with execution server.');
    }
  };

  const triggerResetKillSwitch = async () => {
    try {
      const res = await fetch(`${apiUrl}/api/execution/kill-switch/reset?operator=MobileApp`, {
        method: 'POST',
      });
      if (res.ok) {
        await loadShadowData();
        Alert.alert('Kill Switch Reset', 'Trading execution restored to active state.');
      }
    } catch (err) {
      Alert.alert('Error', 'Failed to communicate with execution server.');
    }
  };

  // Initial load
  useEffect(() => {
    loadSignalsData();
  }, [loadSignalsData]);

  useEffect(() => {
    if (activeTab === 'movers') {
      loadMoversData();
    } else if (activeTab === 'history') {
      loadHistoryData();
    } else if (activeTab === 'shadow') {
      loadShadowData();
    }
  }, [activeTab, loadMoversData, loadHistoryData, loadShadowData]);

  // Live ML context sync for Deep-Dive inspection
  useEffect(() => {
    if (!selectedStock) {
      setSelectedStockML(null);
      return;
    }
    const sym = (selectedStock.symbol || selectedStock.stock || '').trim().toUpperCase().split('.')[0];
    if (!sym) return;

    let isMounted = true;
    setLoadingML(true);
    fetch(`${apiUrl}/api/ml/context/${sym}`)
      .then((res) => (res.ok ? res.json() : null))
      .then((data) => {
        if (isMounted) {
          if (data && data.available && data.data) {
            setSelectedStockML(data.data);
          } else {
            setSelectedStockML(null);
          }
          setLoadingML(false);
        }
      })
      .catch(() => {
        if (isMounted) {
          setSelectedStockML(null);
          setLoadingML(false);
        }
      });

    return () => {
      isMounted = false;
    };
  }, [selectedStock, apiUrl]);

  const onRefresh = async () => {
    setRefreshing(true);
    if (activeTab === 'signals') await loadSignalsData();
    else if (activeTab === 'movers') await loadMoversData();
    else if (activeTab === 'history') await loadHistoryData();
    else if (activeTab === 'shadow') await loadShadowData();
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
          model: 'flint',
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
              await loadSignalsData();
              try {
                const targetUpper = symbol.trim().toUpperCase();
                const checkRes = await fetch(`${apiUrl}/api/signals/latest?days=7&search=${targetUpper}`);
                if (checkRes.ok) {
                  const checkData = await checkRes.json();
                  if (checkData.signals && checkData.signals.length > 0) {
                    setSelectedStock(checkData.signals[0]);
                  }
                }
              } catch (e) {}
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

  // Filtered & Sorted Signals (Sorted by date the analysis executed)
  const filteredSignals = useMemo(() => {
    return latestSignals
      .filter((s) => {
        const matchesDecision = signalDecisionFilter === 'ALL' || s.decision === signalDecisionFilter;
        const query = signalSearch.trim().toUpperCase();
        const stockName = (s.company_name || getStockName(s) || '').toUpperCase();
        const matchesSearch =
          !query || s.symbol.toUpperCase().includes(query) || stockName.includes(query);
        return matchesDecision && matchesSearch;
      })
      .sort((a, b) => {
        const timeA = a.timestamp ? new Date(a.timestamp.replace(' ', 'T')).getTime() || 0 : 0;
        const timeB = b.timestamp ? new Date(b.timestamp.replace(' ', 'T')).getTime() || 0 : 0;
        return signalSortOrder === 'asc' ? timeA - timeB : timeB - timeA;
      });
  }, [latestSignals, signalDecisionFilter, signalSearch, signalSortOrder]);

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
                placeholder="Search stock name or symbol (e.g. Apple, NVDA)..."
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

            {/* Signal List Cards Header with Execution Date Sort Toggle */}
            <View style={styles.sectionHeadingRow}>
              <Text style={[styles.sectionHeading, { marginTop: 0, marginBottom: 0 }]}>
                Active Recommendations ({filteredSignals.length})
              </Text>
              <TouchableOpacity
                style={styles.sortToggleBtn}
                onPress={() => setSignalSortOrder((prev) => (prev === 'desc' ? 'asc' : 'desc'))}
                accessibilityRole="button"
                accessibilityLabel="Toggle execution date sort order"
              >
                <Text style={styles.sortToggleText}>
                  {signalSortOrder === 'desc' ? '📅 Date: Newest ↓' : '📅 Date: Oldest ↑'}
                </Text>
              </TouchableOpacity>
            </View>

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
                const mlInfo = getMLInfo(item);

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
                      <View style={styles.signalTitleContainer}>
                        <Text style={styles.signalSymbol} numberOfLines={1}>
                          {getStockName(item)}
                        </Text>
                        <Text style={styles.signalModel}>{item.model_used || '6-Agent AI'}</Text>
                      </View>
                      <View style={[styles.badge, { backgroundColor: badgeColor }]}>
                        <Text style={[styles.badgeText, { color: textColor }]}>{item.decision}</Text>
                      </View>
                    </View>

                    {/* ML Pipeline Standalone Indicator Banner */}
                    <View
                      style={[
                        styles.signalMLBanner,
                        {
                          backgroundColor: mlInfo.available ? mlInfo.bgColor : '#0E131F',
                          borderColor: mlInfo.available ? mlInfo.color : '#212C3D',
                        },
                      ]}
                    >
                      <View style={styles.signalMLBannerLeft}>
                        <Text style={styles.signalMLTitle}>🤖 ML Alpha</Text>
                        <View style={[styles.mlBadgePill, { borderColor: mlInfo.color }]}>
                          <Text style={[styles.mlBadgePillText, { color: mlInfo.textColor }]}>
                            {mlInfo.available ? mlInfo.label : 'Non-S&P 500'}
                          </Text>
                        </View>
                      </View>
                      <View style={styles.signalMLBannerRight}>
                        {mlInfo.available ? (
                          <Text style={styles.signalMLMetaText}>
                            {mlInfo.rank ? `#${mlInfo.rank}/${mlInfo.universeSize} • ` : ''}
                            <Text style={{ color: (mlInfo.predictedReturn ?? 0) >= 0 ? '#34D399' : '#F87171', fontWeight: '700' }}>
                              {mlInfo.predictedReturnFormatted}
                            </Text>
                          </Text>
                        ) : (
                          <Text style={styles.signalMLMetaTextMuted}>Composite Quant Only</Text>
                        )}
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
                        <Text style={styles.statLabel}>ML Score</Text>
                        <Text style={[styles.statVal, mlInfo.available && { color: mlInfo.color }]}>
                          {mlInfo.available ? `${Math.round(mlInfo.score)}/100` : 'N/A'}
                        </Text>
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

                    {/* Structured Policy Action Row */}
                    {(() => {
                      const sInfo = getStructuredDecisionInfo(item);
                      if (!sInfo || !sInfo.positionAction || sInfo.positionAction === 'NO_ACTION') return null;
                      const isLong = sInfo.positionAction === 'OPEN_LONG';
                      return (
                        <View style={styles.signalPolicyRow}>
                          <Text style={styles.signalPolicyLabel}>Policy Action:</Text>
                          <View
                            style={[
                              styles.signalPolicyBadge,
                              { backgroundColor: isLong ? '#064E3B' : '#1E293B' },
                            ]}
                          >
                            <Text
                              style={[
                                styles.signalPolicyBadgeText,
                                { color: isLong ? '#34D399' : '#CBD5E1' },
                              ]}
                            >
                              {sInfo.positionAction}
                            </Text>
                          </View>
                          {sInfo.sizing && sInfo.sizing.shares_allowed ? (
                            <Text style={styles.signalPolicySizing}>
                              • Sizing: {sInfo.sizing.shares_allowed} shs
                            </Text>
                          ) : null}
                        </View>
                      );
                    })()}

                    {item.reason ? (
                      <Text style={styles.signalReason} numberOfLines={2}>
                        {item.reason}
                      </Text>
                    ) : null}

                    <View style={styles.cardFooter}>
                      <Text style={styles.cardTimestamp}>🕒 Analyzed: {item.timestamp}</Text>
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

              {/* Real-time Stock Name Lookup Preview */}
              {inspectQuery.trim().length > 0 && (
                <View style={styles.lookupPreviewRow}>
                  <Text style={styles.lookupPreviewLabel}>Stock Name:</Text>
                  <Text style={styles.lookupPreviewName}>{getStockName(inspectQuery)}</Text>
                  {(() => {
                    const cleanQ = inspectQuery.trim().toUpperCase();
                    const match =
                      latestSignals.find((s) => (s.symbol || s.stock || '').toUpperCase() === cleanQ) ||
                      historyRecords.find((s) => (s.symbol || s.stock || '').toUpperCase() === cleanQ);
                    if (match) {
                      return (
                        <TouchableOpacity
                          style={styles.lookupSelectBtn}
                          onPress={() => setSelectedStock(match)}
                        >
                          <Text style={styles.lookupSelectBtnText}>View Analysis</Text>
                        </TouchableOpacity>
                      );
                    }
                    return null;
                  })()}
                </View>
              )}

              {analysisStatus ? (
                <View style={styles.statusBox}>
                  <ActivityIndicator size="small" color="#3B82F6" />
                  <Text style={styles.statusBoxText}>{analysisStatus}</Text>
                </View>
              ) : null}
            </View>

            {/* Quick Stock Switcher in Deep-Dive */}
            {latestSignals.length > 0 && (
              <View style={styles.stockSelectorSection}>
                <Text style={styles.stockSelectorLabel}>Select Stock to Inspect:</Text>
                <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.stockSelectorScroll}>
                  {latestSignals.map((item) => {
                    const isSelected =
                      selectedStock &&
                      (selectedStock.symbol || selectedStock.stock) === (item.symbol || item.stock);
                    return (
                      <TouchableOpacity
                        key={item.id || item.symbol}
                        style={[styles.stockPill, isSelected && styles.stockPillActive]}
                        onPress={() => setSelectedStock(item)}
                      >
                        <Text style={[styles.stockPillText, isSelected && styles.stockPillTextActive]}>
                          {getStockName(item)}
                        </Text>
                      </TouchableOpacity>
                    );
                  })}
                </ScrollView>
              </View>
            )}

            {/* Selected Stock Inspection */}
            {selectedStock ? (
              <View style={styles.detailContainer}>
                <View style={styles.detailHeader}>
                  <View style={styles.signalTitleContainer}>
                    <Text style={styles.detailTitle}>{getStockName(selectedStock)}</Text>
                    <Text style={styles.detailSubtitle}>
                      {selectedStock.symbol || selectedStock.stock} • Model: {selectedStock.model_used || '6-Agent Ensemble'}
                    </Text>
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

                {selectedStock.decision === 'HOLD' ? (
                  <View style={styles.narrativeBox}>
                    <Text style={styles.narrativeTitle}>Decision Source</Text>
                    <Text style={styles.narrativeText}>
                      {{
                        MODEL: 'The model chose HOLD.',
                        GATE: 'A deterministic risk gate blocked a directional signal.',
                        PROBABILITY_REALIGNED: 'The model probabilities favored HOLD.',
                        SCHEMA_INVALID: 'The model output was invalid; a safe HOLD was recorded.',
                      }[selectedStock.decision_origin] || 'Not recorded for this signal.'}
                      {selectedStock.no_trade_reason ? ` Reason: ${selectedStock.no_trade_reason}.` : ''}
                    </Text>
                  </View>
                ) : null}

                {/* Stock Identification Banner */}
                <View style={styles.companyBanner}>
                  <View style={styles.companyBannerCol}>
                    <Text style={styles.companyBannerLabel}>Company Name</Text>
                    <Text style={styles.companyBannerName}>{getStockName(selectedStock)}</Text>
                  </View>
                  <View style={styles.companyBannerCol}>
                    <Text style={styles.companyBannerLabel}>Ticker Symbol</Text>
                    <Text style={styles.companyBannerSymbol}>{selectedStock.symbol || selectedStock.stock}</Text>
                  </View>
                  <View style={styles.companyBannerCol}>
                    <Text style={styles.companyBannerLabel}>Analysis Date</Text>
                    <Text style={styles.companyBannerDate}>🕒 {selectedStock.timestamp || 'Latest'}</Text>
                  </View>
                </View>

                {/* Standalone ML Alpha Pipeline Card */}
                {(() => {
                  const mlInfo = getMLInfo(selectedStock, selectedStockML);
                  return (
                    <View style={styles.mlCard}>
                      <View style={styles.mlCardHeader}>
                        <View style={{ flex: 1, marginRight: 8 }}>
                          <Text style={styles.mlCardTitle}>🤖 Standalone ML Alpha Indicator</Text>
                          <Text style={styles.mlCardSubtitle}>Point-in-Time XGBoost 20-Session Alpha Model</Text>
                        </View>
                        <View style={[styles.mlBadgePill, { backgroundColor: mlInfo.bgColor, borderColor: mlInfo.color }]}>
                          <Text style={[styles.mlBadgePillText, { color: mlInfo.textColor }]}>
                            {mlInfo.badge}
                          </Text>
                        </View>
                      </View>

                      {mlInfo.available ? (
                        <>
                          {/* S&P 500 Top-10 Model Basket Callout */}
                          {mlInfo.inTop10 && (
                            <View style={styles.mlTop10Banner}>
                              <Text style={styles.mlTop10Text}>🏆 S&P 500 Top-10 Model Portfolio High-Conviction Basket</Text>
                            </View>
                          )}

                          {/* Primary Score Display & Stance */}
                          <View style={styles.mlScoreDisplayRow}>
                            <View>
                              <Text style={styles.mlScoreLarge}>{mlInfo.scoreFormatted}</Text>
                              <Text style={styles.mlScorePercentileText}>
                                Percentile: {mlInfo.score}% {mlInfo.rank ? `• Rank #${mlInfo.rank} of ${mlInfo.universeSize}` : ''}
                              </Text>
                            </View>
                            <View style={styles.mlStanceTag}>
                              <Text style={styles.mlTierText}>{mlInfo.tier}</Text>
                              <Text style={[styles.mlStanceBadge, { color: mlInfo.color }]}>
                                Stance: {mlInfo.stance}
                              </Text>
                            </View>
                          </View>

                          {/* Visual Quintile Meter Bar */}
                          <View style={styles.mlScoreBarTrack}>
                            <View
                              style={[
                                styles.mlScoreBarFill,
                                {
                                  width: `${Math.min(100, Math.max(0, mlInfo.score))}%`,
                                  backgroundColor: mlInfo.color,
                                },
                              ]}
                            />
                          </View>
                          <View style={styles.mlScoreBarLegend}>
                            <Text style={styles.mlLegendText}>0% Underperform</Text>
                            <Text style={styles.mlLegendText}>50% Neutral</Text>
                            <Text style={styles.mlLegendText}>100% Outperform</Text>
                          </View>

                          {/* 4-Stat Metric Grid */}
                          <View style={styles.mlMetricsGrid}>
                            <View style={styles.mlMetricCol}>
                              <Text style={styles.mlMetricLabel}>20d Predicted Alpha</Text>
                              <Text style={[styles.mlMetricVal, { color: (mlInfo.predictedReturn ?? 0) >= 0 ? '#34D399' : '#F87171' }]}>
                                {mlInfo.predictedReturnFormatted}
                              </Text>
                              <Text style={styles.mlMetricSub}>Excess vs S&P 500</Text>
                            </View>
                            <View style={styles.mlMetricCol}>
                              <Text style={styles.mlMetricLabel}>S&P 500 Rank</Text>
                              <Text style={styles.mlMetricVal}>
                                {mlInfo.rank ? `#${mlInfo.rank}` : 'N/A'}
                              </Text>
                              <Text style={styles.mlMetricSub}>of {mlInfo.universeSize} universe</Text>
                            </View>
                            <View style={styles.mlMetricCol}>
                              <Text style={styles.mlMetricLabel}>Technicals Rank</Text>
                              <Text style={styles.mlMetricVal}>
                                {mlInfo.rankTechnicals ? `#${mlInfo.rankTechnicals}` : 'N/A'}
                              </Text>
                              <Text style={styles.mlMetricSub}>1m momentum/trend</Text>
                            </View>
                            <View style={styles.mlMetricCol}>
                              <Text style={styles.mlMetricLabel}>Fundamentals Rank</Text>
                              <Text style={styles.mlMetricVal}>
                                {mlInfo.rankFundamentals ? `#${mlInfo.rankFundamentals}` : 'N/A'}
                              </Text>
                              <Text style={styles.mlMetricSub}>1m PIT fundamentals</Text>
                            </View>
                          </View>

                          {/* LLM Consensus vs ML Model Disposition Callout */}
                          {(mlInfo.dispositionReason || mlInfo.disposition) && (
                            <View style={styles.mlAssessmentBox}>
                              <View style={styles.mlAssessmentHeader}>
                                <Text style={styles.mlAssessmentTitle}>🤖 6-Agent Consensus vs. ML Disposition</Text>
                                <View
                                  style={[
                                    styles.mlDispositionPill,
                                    {
                                      backgroundColor:
                                        mlInfo.disposition === 'AGREE'
                                          ? '#064E3B'
                                          : mlInfo.disposition === 'DISAGREE'
                                          ? '#7F1D1D'
                                          : '#1E293B',
                                      borderColor:
                                        mlInfo.disposition === 'AGREE'
                                          ? '#10B981'
                                          : mlInfo.disposition === 'DISAGREE'
                                          ? '#EF4444'
                                          : '#64748B',
                                    },
                                  ]}
                                >
                                  <Text
                                    style={[
                                      styles.mlDispositionText,
                                      {
                                        color:
                                          mlInfo.disposition === 'AGREE'
                                            ? '#34D399'
                                            : mlInfo.disposition === 'DISAGREE'
                                            ? '#FCA5A5'
                                            : '#CBD5E1',
                                      },
                                    ]}
                                  >
                                    {mlInfo.disposition}
                                  </Text>
                                </View>
                              </View>
                              {mlInfo.dispositionReason ? (
                                <Text style={styles.mlAssessmentReason}>{mlInfo.dispositionReason}</Text>
                              ) : null}
                              {mlInfo.forecastOutlook && (
                                <Text style={styles.mlForecastText}>
                                  • 20-Day Outlook: <Text style={{ color: '#60A5FA', fontWeight: '700' }}>{mlInfo.forecastOutlook}</Text>
                                  {mlInfo.forecastConfidence ? ` (${mlInfo.forecastConfidence}% conf)` : ''}
                                  {mlInfo.forecastRationale ? ` — ${mlInfo.forecastRationale}` : ''}
                                </Text>
                              )}
                            </View>
                          )}

                          {/* Model Transparency Info */}
                          <View style={styles.mlMetaRow}>
                            <Text style={styles.mlMetaText}>
                              Model: {mlInfo.modelName} • As-Of: {mlInfo.scoreDate || 'Fresh'} • Horizon: 20 Trading Days
                            </Text>
                          </View>
                        </>
                      ) : (
                        <View style={styles.mlUnavailableBox}>
                          <Text style={styles.mlUnavailableTitle}>⚪ Outside S&P 500 Universe</Text>
                          <Text style={styles.mlUnavailableText}>
                            {selectedStock.symbol || selectedStock.stock} is outside the S&P 500 large-cap universe. The point-in-time XGBoost ML model evaluates the US S&P 500 universe (483 constituents). The 6-agent system utilizes the deterministic 5-pillar quantitative model for this setup.
                          </Text>
                        </View>
                      )}
                    </View>
                  );
                })()}

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

                {/* Structured Decision & Policy Compliance Gates Card (Step 6.4 & 7.2) */}
                {(() => {
                  const structInfo = getStructuredDecisionInfo(selectedStock);
                  const action = structInfo?.positionAction || 'NO_ACTION';
                  const actionColor =
                    action === 'OPEN_LONG' ? '#10B981' :
                    action === 'OPEN_SHORT' ? '#EF4444' :
                    action === 'CLOSE_LONG' || action === 'CLOSE_SHORT' ? '#F59E0B' : '#94A3B8';
                  const actionBg =
                    action === 'OPEN_LONG' ? '#064E3B' :
                    action === 'OPEN_SHORT' ? '#7F1D1D' :
                    action === 'CLOSE_LONG' || action === 'CLOSE_SHORT' ? '#78350F' : '#1E293B';

                  const sizing = structInfo?.sizing || {};
                  return (
                    <>
                      <Text style={styles.subHeading}>🛡️ Structured Decision & Policy Gates</Text>
                      <View style={styles.structuredCard}>
                        <View style={styles.structuredHeaderRow}>
                          <View>
                            <Text style={styles.structuredHeaderTitle}>Policy Engine Action</Text>
                            <Text style={styles.structuredHeaderSub}>Deterministic Gates & Policy Rules</Text>
                          </View>
                          <View style={[styles.actionBadge, { backgroundColor: actionBg, borderColor: actionColor }]}>
                            <Text style={[styles.actionBadgeText, { color: actionColor }]}>{action}</Text>
                          </View>
                        </View>

                        {/* Passed & Failed Gates */}
                        <View style={styles.gateSection}>
                          <Text style={styles.gateSectionLabel}>Passed Compliance Gates ({structInfo?.passedGates?.length || 0})</Text>
                          <View style={styles.gatePillsRow}>
                            {structInfo?.passedGates && structInfo.passedGates.length > 0 ? (
                              structInfo.passedGates.map((gate, idx) => (
                                <View key={idx} style={styles.passedGatePill}>
                                  <Text style={styles.passedGateText}>✓ {gate}</Text>
                                </View>
                              ))
                            ) : (
                              <Text style={styles.gateEmptyText}>None recorded</Text>
                            )}
                          </View>
                        </View>

                        {structInfo?.failedGates && structInfo.failedGates.length > 0 && (
                          <View style={styles.gateSection}>
                            <Text style={[styles.gateSectionLabel, { color: '#F87171' }]}>
                              Failed / Restricting Gates ({structInfo.failedGates.length})
                            </Text>
                            <View style={styles.gatePillsRow}>
                              {structInfo.failedGates.map((gate, idx) => (
                                <View key={idx} style={styles.failedGatePill}>
                                  <Text style={styles.failedGateText}>✗ {gate}</Text>
                                </View>
                              ))}
                            </View>
                          </View>
                        )}

                        {/* Programmatic Reason Codes */}
                        {structInfo?.reasonCodes && structInfo.reasonCodes.length > 0 && (
                          <View style={styles.gateSection}>
                            <Text style={styles.gateSectionLabel}>Programmatic Reason Codes</Text>
                            <View style={styles.gatePillsRow}>
                              {structInfo.reasonCodes.map((code, idx) => (
                                <View key={idx} style={styles.reasonCodePill}>
                                  <Text style={styles.reasonCodeText}>🏷️ {code}</Text>
                                </View>
                              ))}
                            </View>
                          </View>
                        )}
                      </View>

                      {/* Institutional Position Sizing & Controls Card (Step 7.1 - 7.3) */}
                      <Text style={styles.subHeading}>📐 Risk-Based Position Sizing</Text>
                      <View style={styles.sizingCard}>
                        <View style={styles.sizingHeaderRow}>
                          <View>
                            <Text style={styles.sizingHeaderTitle}>Fixed-Fractional & Volatility Sizing</Text>
                            <Text style={styles.sizingHeaderSub}>1% Account Risk • ATR-Gap Reserve Buffer</Text>
                          </View>
                          {sizing.binding_constraint && (
                            <View style={styles.constraintBadge}>
                              <Text style={styles.constraintBadgeText}>
                                {String(sizing.binding_constraint).replace(/_/g, ' ')}
                              </Text>
                            </View>
                          )}
                        </View>

                        <View style={styles.sizingGrid}>
                          <View style={styles.sizingCol}>
                            <Text style={styles.sizingLabel}>Shares Allowed</Text>
                            <Text style={[styles.sizingVal, { color: (sizing.shares_allowed || 0) > 0 ? '#34D399' : '#94A3B8' }]}>
                              {sizing.shares_allowed ?? 0} shs
                            </Text>
                            <Text style={styles.sizingSub}>
                              {(sizing.shares_allowed || 0) > 0 ? `~$${Math.round((sizing.shares_allowed || 0) * (selectedStock.entry_price || selectedStock.price || 0))}` : 'Entry barred'}
                            </Text>
                          </View>
                          <View style={styles.sizingCol}>
                            <Text style={styles.sizingLabel}>Risk Budget</Text>
                            <Text style={styles.sizingVal}>
                              ${Math.round(sizing.risk_budget_usd ?? 1000)}
                            </Text>
                            <Text style={styles.sizingSub}>1.0% portfolio equity</Text>
                          </View>
                          <View style={styles.sizingCol}>
                            <Text style={styles.sizingLabel}>Dollar Risk</Text>
                            <Text style={[styles.sizingVal, { color: '#F87171' }]}>
                              ${Math.round(sizing.dollar_risk ?? 0)}
                            </Text>
                            <Text style={styles.sizingSub}>Max stop-out loss</Text>
                          </View>
                          <View style={styles.sizingCol}>
                            <Text style={styles.sizingLabel}>Gap Reserve</Text>
                            <Text style={styles.sizingVal}>
                              {sizing.gap_reserve_pct ? `${Math.round(sizing.gap_reserve_pct * 100)}%` : '20%'}
                            </Text>
                            <Text style={styles.sizingSub}>Overnight gap buffer</Text>
                          </View>
                        </View>

                        {sizing.rejection_reasons && sizing.rejection_reasons.length > 0 && (
                          <View style={styles.sizingRejectBox}>
                            <Text style={styles.sizingRejectTitle}>⚠️ Sizing Restrictions Active:</Text>
                            {sizing.rejection_reasons.map((r, i) => (
                              <Text key={i} style={styles.sizingRejectText}>• {r}</Text>
                            ))}
                          </View>
                        )}
                      </View>
                    </>
                  );
                })()}

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
                <Text style={styles.emptyText}>Select a stock from the Signals tab or choose an analyzed stock below.</Text>
                {latestSignals.length > 0 && (
                  <TouchableOpacity
                    style={[styles.actionBtn, { marginTop: 12 }]}
                    onPress={() => setSelectedStock(latestSignals[0])}
                  >
                    <Text style={styles.actionBtnText}>
                      Inspect {getStockName(latestSignals[0])}
                    </Text>
                  </TouchableOpacity>
                )}
              </View>
            )}
          </View>
        )}

        {/* ==================== TAB: SHADOW TRADING & EXECUTION (Steps 7.3 & 8) ==================== */}
        {activeTab === 'shadow' && (
          <View style={styles.tabContainer}>
            {/* Operational Kill Switch Status Banner */}
            <View
              style={[
                styles.killSwitchBanner,
                {
                  backgroundColor: killSwitchState?.is_tripped ? '#450A0A' : '#052E16',
                  borderColor: killSwitchState?.is_tripped ? '#EF4444' : '#10B981',
                },
              ]}
            >
              <View style={styles.killSwitchHeader}>
                <View style={{ flex: 1, marginRight: 10 }}>
                  <Text
                    style={[
                      styles.killSwitchTitle,
                      { color: killSwitchState?.is_tripped ? '#FCA5A5' : '#86EFAC' },
                    ]}
                  >
                    {killSwitchState?.is_tripped ? '🛑 OPERATIONAL KILL SWITCH: TRIPPED' : '⚡ KILL SWITCH: OPERATIONAL & ACTIVE'}
                  </Text>
                  <Text style={styles.killSwitchDesc}>
                    {killSwitchState?.is_tripped
                      ? `Execution frozen by ${killSwitchState.tripped_by || 'system'}: ${killSwitchState.trip_reason || 'Manual Trip'}`
                      : 'Pre-flight order gating active. Position reconciler and risk monitors operational.'}
                  </Text>
                </View>
                <TouchableOpacity
                  style={[
                    styles.killSwitchBtn,
                    {
                      backgroundColor: killSwitchState?.is_tripped ? '#10B981' : '#EF4444',
                    },
                  ]}
                  onPress={() => {
                    if (killSwitchState?.is_tripped) {
                      triggerResetKillSwitch();
                    } else {
                      Alert.alert(
                        'Trip Kill Switch?',
                        'This will freeze all new risk entries across simulated and production execution.',
                        [
                          { text: 'Cancel', style: 'cancel' },
                          { text: 'TRIP KILL SWITCH', style: 'destructive', onPress: triggerTripKillSwitch },
                        ]
                      );
                    }
                  }}
                >
                  <Text style={styles.killSwitchBtnText}>
                    {killSwitchState?.is_tripped ? 'RESTORE' : 'TRIP'}
                  </Text>
                </TouchableOpacity>
              </View>
            </View>

            {/* Institutional Promotion Criteria Scorecard (Step 8) */}
            <View style={styles.promoCard}>
              <View style={styles.promoHeader}>
                <View>
                  <Text style={styles.promoTitle}>🏆 Institutional Promotion Gate</Text>
                  <Text style={styles.promoSub}>Predefined 8 Formal Verification Gates</Text>
                </View>
                {shadowMetrics?.promotion_decision && (
                  <View
                    style={[
                      styles.promoStatusPill,
                      {
                        backgroundColor:
                          shadowMetrics.promotion_decision.status === 'PASSED'
                            ? '#064E3B'
                            : shadowMetrics.promotion_decision.status === 'PENDING_OBSERVATION'
                            ? '#78350F'
                            : '#7F1D1D',
                        borderColor:
                          shadowMetrics.promotion_decision.status === 'PASSED'
                            ? '#10B981'
                            : shadowMetrics.promotion_decision.status === 'PENDING_OBSERVATION'
                            ? '#F59E0B'
                            : '#EF4444',
                      },
                    ]}
                  >
                    <Text
                      style={[
                        styles.promoStatusText,
                        {
                          color:
                            shadowMetrics.promotion_decision.status === 'PASSED'
                              ? '#34D399'
                              : shadowMetrics.promotion_decision.status === 'PENDING_OBSERVATION'
                              ? '#FCD34D'
                              : '#FCA5A5',
                        },
                      ]}
                    >
                      {shadowMetrics.promotion_decision.status}
                    </Text>
                  </View>
                )}
              </View>

              {shadowMetrics?.promotion_decision ? (
                <>
                  <Text style={styles.promoRationale}>
                    {shadowMetrics.promotion_decision.recommendation_summary || 'Evaluating shadow trading stream against institutional promotion criteria.'}
                  </Text>
                  <View style={styles.promoProgressRow}>
                    <Text style={styles.promoProgressText}>
                      Gates Passed: {shadowMetrics.promotion_decision.passed_requirements_count || 0} / {shadowMetrics.promotion_decision.total_requirements_count || 7}
                    </Text>
                    <Text style={styles.promoProgressText}>
                      Observations: {shadowMetrics.promotion_decision.observation_count || 0} / {shadowMetrics.promotion_decision.required_min_observations || 30}
                    </Text>
                  </View>
                </>
              ) : null}

              {/* 8 Formal Shadow Measurements Grid */}
              {shadowMetrics?.metrics ? (
                <View style={styles.metricPillGrid}>
                  <View style={styles.metricPillCard}>
                    <Text style={styles.metricPillLabel}>1. Signal Reproducibility</Text>
                    <Text style={styles.metricPillVal}>
                      {((shadowMetrics.metrics.reproducibility?.decision_match_rate ?? 1.0) * 100).toFixed(0)}%
                    </Text>
                    <Text style={styles.metricPillSub}>Deterministic match</Text>
                  </View>
                  <View style={styles.metricPillCard}>
                    <Text style={styles.metricPillLabel}>2. Latency & Age</Text>
                    <Text style={styles.metricPillVal}>
                      {((shadowMetrics.metrics.latency_freshness?.median_decision_latency_seconds ?? 0.001) * 1000).toFixed(1)}ms
                    </Text>
                    <Text style={styles.metricPillSub}>
                      Freshness: {shadowMetrics.metrics.latency_freshness?.staleness_violations ?? 0} stale
                    </Text>
                  </View>
                  <View style={styles.metricPillCard}>
                    <Text style={styles.metricPillLabel}>3. Predicted vs Realized</Text>
                    <Text style={styles.metricPillVal}>
                      {((shadowMetrics.metrics.predictive_performance?.directional_accuracy ?? 0) * 100).toFixed(0)}%
                    </Text>
                    <Text style={styles.metricPillSub}>Directional hit rate</Text>
                  </View>
                  <View style={styles.metricPillCard}>
                    <Text style={styles.metricPillLabel}>4. Cost-Adjusted Net</Text>
                    <Text style={styles.metricPillVal}>
                      {(shadowMetrics.metrics.cost_adjusted?.net_sharpe_ratio ?? 0).toFixed(2)}
                    </Text>
                    <Text style={styles.metricPillSub}>
                      Drag: {(shadowMetrics.metrics.cost_adjusted?.slippage_drag_bps ?? 0).toFixed(1)} bps
                    </Text>
                  </View>
                  <View style={styles.metricPillCard}>
                    <Text style={styles.metricPillLabel}>5. Gate Rejection Rate</Text>
                    <Text style={styles.metricPillVal}>
                      {((shadowMetrics.metrics.gate_efficiency?.rejection_rate ?? 0) * 100).toFixed(0)}%
                    </Text>
                    <Text style={styles.metricPillSub}>Risk-screened out</Text>
                  </View>
                  <View style={styles.metricPillCard}>
                    <Text style={styles.metricPillLabel}>6. Model Drift (PSI)</Text>
                    <Text style={styles.metricPillVal}>
                      {(shadowMetrics.metrics.model_drift?.aggregate_psi ?? 0).toFixed(3)}
                    </Text>
                    <Text style={styles.metricPillSub}>Feature stability</Text>
                  </View>
                  <View style={styles.metricPillCard}>
                    <Text style={styles.metricPillLabel}>7. LLM Temporal Integrity</Text>
                    <Text style={[styles.metricPillVal, { color: '#34D399' }]}>
                      {((shadowMetrics.metrics.llm_temporal_integrity?.point_in_time_adherence ?? 1.0) * 100).toFixed(0)}%
                    </Text>
                    <Text style={styles.metricPillSub}>Zero look-ahead bias</Text>
                  </View>
                  <View style={styles.metricPillCard}>
                    <Text style={styles.metricPillLabel}>8. Ledger Reconciliation</Text>
                    <Text style={[styles.metricPillVal, { color: (shadowMetrics.metrics.risk_compliance?.reconciliation_discrepancies ?? 0) === 0 ? '#34D399' : '#EF4444' }]}>
                      {shadowMetrics.metrics.risk_compliance?.reconciliation_discrepancies ?? 0} errors
                    </Text>
                    <Text style={styles.metricPillSub}>State drift count</Text>
                  </View>
                </View>
              ) : null}
            </View>

            {/* Shadow Trading Decision Audit Log */}
            <Text style={styles.sectionHeading}>
              ⚡ Shadow Audit Trail ({shadowDecisions.length})
            </Text>

            {loadingShadow ? (
              <View style={styles.loadingBox}>
                <ActivityIndicator size="large" color="#3B82F6" />
                <Text style={styles.loadingText}>Fetching shadow execution records...</Text>
              </View>
            ) : shadowDecisions.length === 0 ? (
              <View style={styles.emptyCard}>
                <Text style={styles.emptyText}>No shadow trading decisions logged yet.</Text>
                <Text style={[styles.emptyText, { marginTop: 6, fontSize: 11, color: '#64748B' }]}>
                  Run an analysis on any ticker or run the shadow desk to populate simulated execution logs.
                </Text>
              </View>
            ) : (
              shadowDecisions.map((dec, idx) => {
                const isBuy = dec.decision === 'BUY';
                const isSell = dec.decision === 'SELL';
                const badgeColor = isBuy ? '#059669' : isSell ? '#DC2626' : '#4B5563';
                const textColor = isBuy ? '#34D399' : isSell ? '#F87171' : '#D1D5DB';
                const passedGates = dec.passed_gates || [];
                const failedGates = dec.failed_gates || [];
                const reasonCodes = dec.reason_codes || [];

                return (
                  <View key={dec.run_id || idx} style={styles.shadowCard}>
                    <View style={styles.shadowHeaderRow}>
                      <View>
                        <Text style={styles.shadowTicker}>{dec.ticker}</Text>
                        <Text style={styles.shadowActionText}>Action: {dec.position_action || 'NO_ACTION'}</Text>
                      </View>
                      <View style={[styles.badge, { backgroundColor: badgeColor }]}>
                        <Text style={[styles.badgeText, { color: textColor }]}>{dec.decision}</Text>
                      </View>
                    </View>

                    {/* Sizing & Execution Stats */}
                    <View style={styles.shadowStatsRow}>
                      <View style={styles.shadowStatCol}>
                        <Text style={styles.shadowStatLabel}>Composite</Text>
                        <Text style={styles.shadowStatVal}>{dec.composite_score ? dec.composite_score.toFixed(1) : 'N/A'}</Text>
                      </View>
                      <View style={styles.shadowStatCol}>
                        <Text style={styles.shadowStatLabel}>Proposed Shs</Text>
                        <Text style={styles.shadowStatVal}>{dec.proposed_shares ?? 0}</Text>
                      </View>
                      <View style={styles.shadowStatCol}>
                        <Text style={styles.shadowStatLabel}>Sim Fill</Text>
                        <Text style={styles.shadowStatVal}>
                          {dec.simulated_fill_price ? `$${dec.simulated_fill_price.toFixed(2)}` : 'N/A'}
                        </Text>
                      </View>
                      <View style={styles.shadowStatCol}>
                        <Text style={styles.shadowStatLabel}>Slippage</Text>
                        <Text style={styles.shadowStatVal}>
                          {dec.simulated_slippage_bps ? `${dec.simulated_slippage_bps.toFixed(1)} bps` : '0 bps'}
                        </Text>
                      </View>
                    </View>

                    {/* Gates & Reason Codes Pills */}
                    {(passedGates.length > 0 || failedGates.length > 0 || reasonCodes.length > 0) && (
                      <View style={styles.shadowGatesBox}>
                        <View style={styles.gatePillsRow}>
                          {passedGates.slice(0, 3).map((g, i) => (
                            <View key={i} style={styles.passedGatePill}>
                              <Text style={styles.passedGateText}>✓ {g}</Text>
                            </View>
                          ))}
                          {failedGates.slice(0, 3).map((g, i) => (
                            <View key={i} style={styles.failedGatePill}>
                              <Text style={styles.failedGateText}>✗ {g}</Text>
                            </View>
                          ))}
                          {reasonCodes.slice(0, 2).map((rc, i) => (
                            <View key={i} style={styles.reasonCodePill}>
                              <Text style={styles.reasonCodeText}>🏷️ {rc}</Text>
                            </View>
                          ))}
                        </View>
                      </View>
                    )}

                    <View style={styles.cardFooter}>
                      <Text style={styles.cardTimestamp}>🕒 {dec.timestamp}</Text>
                      <Text style={styles.shadowStatusText}>
                        Status: {dec.simulated_status || 'SIMULATED'}
                      </Text>
                    </View>
                  </View>
                );
              })
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
                      <Text style={styles.moverActionBtnText}>⚡ Analyze with Ling 3.1</Text>
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
              placeholder="Search historical stock name or symbol..."
              placeholderTextColor="#64748B"
              value={historySearch}
              onChangeText={setHistorySearch}
            />
            <Text style={styles.sectionHeading}>Past Signals Log ({historyRecords.length})</Text>

            {historyRecords
              .filter((r) => {
                const q = historySearch.trim().toUpperCase();
                const name = getStockName(r).toUpperCase();
                return !q || r.symbol.toUpperCase().includes(q) || name.includes(q);
              })
              .map((rec) => (
                <View key={rec.id} style={styles.historyCard}>
                  <View style={styles.signalHeader}>
                    <View style={styles.signalTitleContainer}>
                      <Text style={styles.signalSymbol} numberOfLines={1}>{getStockName(rec)}</Text>
                      <Text style={styles.signalModel}>{rec.model_used || '6-Agent AI'}</Text>
                    </View>
                    <Text style={styles.historyDec}>{rec.decision}</Text>
                  </View>
                  {(() => {
                    const hml = getMLInfo(rec);
                    return (
                      <View style={styles.historyMLRow}>
                        <Text style={styles.historyMLLabel}>🤖 ML Alpha:</Text>
                        <Text style={[styles.historyMLVal, { color: hml.color }]}>
                          {hml.available ? `${hml.scoreFormatted} (${hml.label})` : 'Non-S&P 500'}
                        </Text>
                        {hml.available && hml.rank ? (
                          <Text style={styles.historyMLRank}>• Rank #{hml.rank}/{hml.universeSize}</Text>
                        ) : null}
                      </View>
                    );
                  })()}
                  <Text style={styles.historyReason} numberOfLines={2}>{rec.reason}</Text>
                  <Text style={styles.cardTimestamp}>🕒 Analyzed: {rec.timestamp}</Text>
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
              <View style={{ flexDirection: 'row', flexWrap: 'wrap', gap: 8, marginVertical: 8 }}>
                {[
                  { label: '🍓 Raspberry Pi', url: 'http://192.168.0.92:8000' },
                  { label: '💻 Workstation', url: 'http://192.168.0.241:8000' },
                  { label: '📱 Localhost', url: 'http://127.0.0.1:8000' },
                ].map((preset) => (
                  <TouchableOpacity
                    key={preset.url}
                    style={[
                      styles.filterChip,
                      apiUrl === preset.url && styles.filterChipActive,
                      { paddingHorizontal: 10, paddingVertical: 6 },
                    ]}
                    onPress={() => {
                      setApiUrl(preset.url);
                    }}
                  >
                    <Text
                      style={[
                        styles.filterChipText,
                        apiUrl === preset.url && styles.filterChipTextActive,
                      ]}
                    >
                      {preset.label}
                    </Text>
                  </TouchableOpacity>
                ))}
              </View>
              <TouchableOpacity
                style={styles.actionBtn}
                onPress={async () => {
                  try {
                    const res = await fetch(`${apiUrl}/api/health`);
                    const data = await res.json();
                    if (res.ok) {
                      setConnected(true);
                      await loadSignalsData();
                      Alert.alert('✅ Connected!', `API reachable at ${apiUrl} (${data.version})`);
                    }
                  } catch (e) {
                    setConnected(false);
                    Alert.alert('❌ Connection Failed', `Could not reach ${apiUrl}`);
                  }
                }}
              >
                <Text style={styles.actionBtnText}>Test Server Connection & Refresh</Text>
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
          { key: 'shadow', label: 'Execution', icon: '⚡' },
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
  sectionHeadingRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginTop: 6,
    marginBottom: 10,
  },
  sectionHeading: {
    fontSize: 15,
    fontWeight: '700',
    color: '#E2E8F0',
    marginTop: 6,
    marginBottom: 10,
  },
  sortToggleBtn: {
    paddingHorizontal: 10,
    paddingVertical: 5,
    backgroundColor: '#1E293B',
    borderRadius: 6,
    borderWidth: 1,
    borderColor: '#334155',
  },
  sortToggleText: {
    fontSize: 12,
    fontWeight: '600',
    color: '#60A5FA',
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
  signalTitleContainer: {
    flex: 1,
    marginRight: 10,
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
  // ML Banner in Signal Cards
  signalMLBanner: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: 8,
    paddingVertical: 5,
    borderRadius: 6,
    borderWidth: 1,
    marginBottom: 8,
  },
  signalMLBannerLeft: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
  },
  signalMLTitle: {
    fontSize: 11,
    fontWeight: '700',
    color: '#94A3B8',
  },
  mlBadgePill: {
    paddingHorizontal: 6,
    paddingVertical: 2,
    borderRadius: 4,
    borderWidth: 1,
  },
  mlBadgePillText: {
    fontSize: 10,
    fontWeight: '800',
    letterSpacing: 0.3,
  },
  signalMLBannerRight: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  signalMLMetaText: {
    fontSize: 11,
    color: '#94A3B8',
    fontWeight: '600',
  },
  signalMLMetaTextMuted: {
    fontSize: 10,
    color: '#64748B',
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
  lookupPreviewRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: 8,
    paddingTop: 8,
    borderTopWidth: 1,
    borderTopColor: '#212C3D',
    flexWrap: 'wrap',
    gap: 8,
  },
  lookupPreviewLabel: {
    fontSize: 12,
    color: '#94A3B8',
  },
  lookupPreviewName: {
    fontSize: 13,
    fontWeight: '700',
    color: '#60A5FA',
    flex: 1,
  },
  lookupSelectBtn: {
    backgroundColor: '#2563EB',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 4,
  },
  lookupSelectBtnText: {
    fontSize: 11,
    fontWeight: '700',
    color: '#FFFFFF',
  },
  stockSelectorSection: {
    marginBottom: 12,
  },
  stockSelectorLabel: {
    fontSize: 12,
    color: '#94A3B8',
    marginBottom: 6,
    fontWeight: '600',
  },
  stockSelectorScroll: {
    flexDirection: 'row',
  },
  stockPill: {
    paddingHorizontal: 12,
    paddingVertical: 7,
    backgroundColor: '#151B26',
    borderRadius: 6,
    borderWidth: 1,
    borderColor: '#212C3D',
    marginRight: 8,
  },
  stockPillActive: {
    backgroundColor: '#1E293B',
    borderColor: '#3B82F6',
  },
  stockPillText: {
    fontSize: 12,
    color: '#94A3B8',
    fontWeight: '600',
  },
  stockPillTextActive: {
    color: '#60A5FA',
    fontWeight: '700',
  },
  companyBanner: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    backgroundColor: '#0E131F',
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#1E293B',
    padding: 10,
    marginBottom: 12,
  },
  companyBannerCol: {
    flex: 1,
  },
  companyBannerLabel: {
    fontSize: 10,
    color: '#64748B',
    marginBottom: 2,
    textTransform: 'uppercase',
  },
  companyBannerName: {
    fontSize: 13,
    fontWeight: '700',
    color: '#F8FAFC',
  },
  companyBannerSymbol: {
    fontSize: 13,
    fontWeight: '700',
    color: '#38BDF8',
  },
  companyBannerDate: {
    fontSize: 11,
    color: '#94A3B8',
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
  // Dedicated ML Card in Deep Dive
  mlCard: {
    backgroundColor: '#0E131F',
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#1E293B',
    padding: 12,
    marginBottom: 12,
  },
  mlCardHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    marginBottom: 10,
  },
  mlCardTitle: {
    fontSize: 14,
    fontWeight: '800',
    color: '#F8FAFC',
  },
  mlCardSubtitle: {
    fontSize: 10,
    color: '#64748B',
    marginTop: 2,
  },
  mlTop10Banner: {
    backgroundColor: '#064E3B',
    borderWidth: 1,
    borderColor: '#059669',
    borderRadius: 6,
    paddingVertical: 4,
    paddingHorizontal: 8,
    marginBottom: 10,
    alignItems: 'center',
  },
  mlTop10Text: {
    fontSize: 11,
    fontWeight: '700',
    color: '#6EE7B7',
  },
  mlScoreDisplayRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-end',
    marginBottom: 8,
  },
  mlScoreLarge: {
    fontSize: 24,
    fontWeight: '900',
    color: '#F8FAFC',
  },
  mlScorePercentileText: {
    fontSize: 11,
    color: '#94A3B8',
    marginTop: 2,
  },
  mlStanceTag: {
    alignItems: 'flex-end',
  },
  mlTierText: {
    fontSize: 11,
    color: '#CBD5E1',
    fontWeight: '600',
  },
  mlStanceBadge: {
    fontSize: 11,
    fontWeight: '800',
    marginTop: 2,
  },
  mlScoreBarTrack: {
    height: 8,
    backgroundColor: '#1E293B',
    borderRadius: 4,
    overflow: 'hidden',
    marginBottom: 4,
  },
  mlScoreBarFill: {
    height: '100%',
    borderRadius: 4,
  },
  mlScoreBarLegend: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 12,
  },
  mlLegendText: {
    fontSize: 9,
    color: '#64748B',
  },
  mlMetricsGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 6,
    marginBottom: 10,
  },
  mlMetricCol: {
    flex: 1,
    minWidth: '45%',
    backgroundColor: '#151B26',
    borderRadius: 6,
    padding: 8,
    borderWidth: 1,
    borderColor: '#212C3D',
  },
  mlMetricLabel: {
    fontSize: 10,
    color: '#64748B',
  },
  mlMetricVal: {
    fontSize: 14,
    fontWeight: '800',
    color: '#F1F5F9',
    marginTop: 2,
  },
  mlMetricSub: {
    fontSize: 9,
    color: '#94A3B8',
    marginTop: 2,
  },
  mlAssessmentBox: {
    backgroundColor: '#151B26',
    borderRadius: 6,
    borderWidth: 1,
    borderColor: '#212C3D',
    padding: 10,
    marginBottom: 10,
  },
  mlAssessmentHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 6,
  },
  mlAssessmentTitle: {
    fontSize: 11,
    fontWeight: '700',
    color: '#E2E8F0',
    flex: 1,
  },
  mlDispositionPill: {
    paddingHorizontal: 8,
    paddingVertical: 2,
    borderRadius: 4,
    borderWidth: 1,
  },
  mlDispositionText: {
    fontSize: 10,
    fontWeight: '800',
  },
  mlAssessmentReason: {
    fontSize: 11,
    color: '#94A3B8',
    lineHeight: 16,
    marginBottom: 4,
  },
  mlForecastText: {
    fontSize: 11,
    color: '#CBD5E1',
    lineHeight: 16,
    marginTop: 4,
  },
  mlMetaRow: {
    borderTopWidth: 1,
    borderTopColor: '#1F2937',
    paddingTop: 6,
  },
  mlMetaText: {
    fontSize: 10,
    color: '#64748B',
  },
  mlUnavailableBox: {
    paddingVertical: 8,
  },
  mlUnavailableTitle: {
    fontSize: 12,
    fontWeight: '700',
    color: '#94A3B8',
    marginBottom: 4,
  },
  mlUnavailableText: {
    fontSize: 11,
    color: '#64748B',
    lineHeight: 16,
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
  historyMLRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    marginVertical: 4,
  },
  historyMLLabel: {
    fontSize: 11,
    color: '#64748B',
  },
  historyMLVal: {
    fontSize: 11,
    fontWeight: '700',
  },
  historyMLRank: {
    fontSize: 10,
    color: '#64748B',
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
  // Structured Decisions & Policy Sizing (Steps 6.4, 7.1, 7.2)
  signalPolicyRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: 6,
    paddingTop: 6,
    borderTopWidth: 1,
    borderTopColor: '#1E293B',
  },
  signalPolicyLabel: {
    fontSize: 11,
    color: '#94A3B8',
    fontWeight: '600',
    marginRight: 6,
  },
  signalPolicyBadge: {
    paddingHorizontal: 6,
    paddingVertical: 2,
    borderRadius: 4,
  },
  signalPolicyBadgeText: {
    fontSize: 10,
    fontWeight: '700',
  },
  signalPolicySizing: {
    fontSize: 10,
    color: '#60A5FA',
    marginLeft: 6,
    fontWeight: '600',
  },
  structuredCard: {
    backgroundColor: '#151B26',
    borderRadius: 10,
    borderWidth: 1,
    borderColor: '#212C3D',
    padding: 14,
    marginBottom: 14,
  },
  structuredHeaderRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 10,
  },
  structuredHeaderTitle: {
    fontSize: 13,
    fontWeight: '700',
    color: '#F1F5F9',
  },
  structuredHeaderSub: {
    fontSize: 11,
    color: '#64748B',
    marginTop: 1,
  },
  actionBadge: {
    paddingHorizontal: 8,
    paddingVertical: 3,
    borderRadius: 6,
    borderWidth: 1,
  },
  actionBadgeText: {
    fontSize: 11,
    fontWeight: '800',
  },
  gateSection: {
    marginTop: 8,
  },
  gateSectionLabel: {
    fontSize: 11,
    fontWeight: '600',
    color: '#94A3B8',
    marginBottom: 4,
  },
  gatePillsRow: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 6,
  },
  passedGatePill: {
    backgroundColor: '#064E3B',
    borderColor: '#10B981',
    borderWidth: 1,
    borderRadius: 4,
    paddingHorizontal: 8,
    paddingVertical: 2,
  },
  passedGateText: {
    color: '#34D399',
    fontSize: 10,
    fontWeight: '600',
  },
  failedGatePill: {
    backgroundColor: '#7F1D1D',
    borderColor: '#EF4444',
    borderWidth: 1,
    borderRadius: 4,
    paddingHorizontal: 8,
    paddingVertical: 2,
  },
  failedGateText: {
    color: '#FCA5A5',
    fontSize: 10,
    fontWeight: '600',
  },
  reasonCodePill: {
    backgroundColor: '#1E293B',
    borderColor: '#334155',
    borderWidth: 1,
    borderRadius: 4,
    paddingHorizontal: 8,
    paddingVertical: 2,
  },
  reasonCodeText: {
    color: '#93C5FD',
    fontSize: 10,
    fontWeight: '600',
  },
  gateEmptyText: {
    color: '#64748B',
    fontSize: 11,
    fontStyle: 'italic',
  },
  sizingCard: {
    backgroundColor: '#151B26',
    borderRadius: 10,
    borderWidth: 1,
    borderColor: '#212C3D',
    padding: 14,
    marginBottom: 14,
  },
  sizingHeaderRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 10,
  },
  sizingHeaderTitle: {
    fontSize: 13,
    fontWeight: '700',
    color: '#F1F5F9',
  },
  sizingHeaderSub: {
    fontSize: 11,
    color: '#64748B',
    marginTop: 1,
  },
  constraintBadge: {
    backgroundColor: '#1E293B',
    borderColor: '#475569',
    borderWidth: 1,
    borderRadius: 4,
    paddingHorizontal: 8,
    paddingVertical: 3,
  },
  constraintBadgeText: {
    color: '#CBD5E1',
    fontSize: 10,
    fontWeight: '700',
  },
  sizingGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
    marginTop: 6,
  },
  sizingCol: {
    width: '48%',
    backgroundColor: '#0D111A',
    borderRadius: 8,
    padding: 10,
    borderWidth: 1,
    borderColor: '#1F2937',
  },
  sizingLabel: {
    fontSize: 10,
    color: '#94A3B8',
    fontWeight: '600',
  },
  sizingVal: {
    fontSize: 16,
    fontWeight: '800',
    color: '#F1F5F9',
    marginTop: 2,
  },
  sizingSub: {
    fontSize: 10,
    color: '#64748B',
    marginTop: 2,
  },
  sizingRejectBox: {
    backgroundColor: '#450A0A',
    borderColor: '#7F1D1D',
    borderWidth: 1,
    borderRadius: 6,
    padding: 8,
    marginTop: 10,
  },
  sizingRejectTitle: {
    color: '#F87171',
    fontSize: 11,
    fontWeight: '700',
    marginBottom: 2,
  },
  sizingRejectText: {
    color: '#FCA5A5',
    fontSize: 10,
    marginTop: 1,
  },
  // Operational Kill Switch & Shadow Promotion (Steps 7.3 & 8)
  killSwitchBanner: {
    borderRadius: 10,
    borderWidth: 1,
    padding: 12,
    marginBottom: 14,
  },
  killSwitchHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  killSwitchTitle: {
    fontSize: 12,
    fontWeight: '800',
  },
  killSwitchDesc: {
    fontSize: 10,
    color: '#CBD5E1',
    marginTop: 2,
    lineHeight: 14,
  },
  killSwitchBtn: {
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 6,
  },
  killSwitchBtnText: {
    color: '#FFFFFF',
    fontSize: 11,
    fontWeight: '800',
  },
  promoCard: {
    backgroundColor: '#151B26',
    borderRadius: 10,
    borderWidth: 1,
    borderColor: '#212C3D',
    padding: 14,
    marginBottom: 14,
  },
  promoHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 6,
  },
  promoTitle: {
    fontSize: 13,
    fontWeight: '700',
    color: '#F1F5F9',
  },
  promoSub: {
    fontSize: 11,
    color: '#64748B',
    marginTop: 1,
  },
  promoStatusPill: {
    paddingHorizontal: 8,
    paddingVertical: 2,
    borderRadius: 6,
    borderWidth: 1,
  },
  promoStatusText: {
    fontSize: 10,
    fontWeight: '800',
  },
  promoRationale: {
    fontSize: 11,
    color: '#94A3B8',
    lineHeight: 16,
    marginTop: 4,
  },
  promoProgressRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginTop: 8,
    paddingVertical: 6,
    borderTopWidth: 1,
    borderTopColor: '#212C3D',
  },
  promoProgressText: {
    fontSize: 11,
    color: '#60A5FA',
    fontWeight: '600',
  },
  metricPillGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
    marginTop: 10,
  },
  metricPillCard: {
    width: '48%',
    backgroundColor: '#0D111A',
    borderRadius: 8,
    padding: 8,
    borderWidth: 1,
    borderColor: '#1F2937',
  },
  metricPillLabel: {
    fontSize: 10,
    color: '#94A3B8',
    fontWeight: '600',
  },
  metricPillVal: {
    fontSize: 14,
    fontWeight: '800',
    color: '#F1F5F9',
    marginTop: 2,
  },
  metricPillSub: {
    fontSize: 9,
    color: '#64748B',
    marginTop: 2,
  },
  shadowCard: {
    backgroundColor: '#151B26',
    borderRadius: 10,
    borderWidth: 1,
    borderColor: '#212C3D',
    padding: 12,
    marginBottom: 10,
  },
  shadowHeaderRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  shadowTicker: {
    fontSize: 14,
    fontWeight: '800',
    color: '#F1F5F9',
  },
  shadowActionText: {
    fontSize: 10,
    color: '#60A5FA',
    fontWeight: '600',
    marginTop: 1,
  },
  shadowStatsRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    backgroundColor: '#0D111A',
    borderRadius: 6,
    padding: 8,
    marginTop: 4,
  },
  shadowStatCol: {
    alignItems: 'center',
  },
  shadowStatLabel: {
    fontSize: 9,
    color: '#64748B',
  },
  shadowStatVal: {
    fontSize: 11,
    fontWeight: '700',
    color: '#E2E8F0',
    marginTop: 2,
  },
  shadowGatesBox: {
    marginTop: 8,
  },
  shadowStatusText: {
    fontSize: 10,
    color: '#94A3B8',
    fontWeight: '600',
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
