// Constraints (ensure uniqueness)
CREATE CONSTRAINT company_ticker IF NOT EXISTS FOR (c:Company) REQUIRE c.ticker IS UNIQUE;
CREATE CONSTRAINT risk_id IF NOT EXISTS FOR (r:Risk) REQUIRE r.risk_id IS UNIQUE;

// Indexes (speed up queries)
CREATE INDEX company_sector IF NOT EXISTS FOR (c:Company) ON (c.sector);
CREATE INDEX risk_category IF NOT EXISTS FOR (r:Risk) ON (r.category);
CREATE INDEX metric_name IF NOT EXISTS FOR (m:Metric) ON (m.metric_name);
CREATE INDEX geo_country IF NOT EXISTS FOR (g:Geography) ON (g.country);