EXPERIMENT_SUMMARY_VIEW = """
-- 1) Create the view (or replace if it already exists)
DROP VIEW IF EXISTS experiment_summary_v2 CASCADE;
CREATE OR REPLACE VIEW experiment_summary_v2 AS
WITH traj_counts AS (
   SELECT
     trajectories.experiment_id,
     count(*) AS cnt
   FROM trajectories
   GROUP BY trajectories.experiment_id
)
SELECT
   e.id,
   e.name,
   e.username,
   e.created_at,
   (e.config->'engine_options'->'sampling_params'->>'model') AS model,
   e.description,
   COALESCE(
     e.config->'env_args'->'attack'->>'attack_id',
     e.config->'env_args'->'attack'->>'type',
     e.config->'env_args'->>'attack'
   ) AS attack_id,
   COALESCE(
     e.config->'env_args'->'defense'->>'defense_id',
     e.config->'env_args'->'defense'->>'type',
     e.config->'env_args'->>'defense'
   ) AS defense_id,
   COALESCE(
     e.config->'env_args'->'defense_kwargs'->'sampling_params'->>'model',
     'None'
   ) AS defense_model,
   COALESCE(
     e.config->'env_args'->'defense_kwargs'->'sampling_params_labeler'->>'model',
     'None'
   ) AS defense_labeler_model,
   COALESCE(td.name, 'Unknown') AS dataset_name,
   COALESCE(tc.cnt, 0)   AS trajectory_count,
   -- TSR: Trajectory Success Rate
   CASE WHEN COALESCE(tc.cnt, 0) = 0 THEN 0
        ELSE ROUND(
            (
              (SELECT COUNT(*)
               FROM trajectories t2
               WHERE t2.experiment_id = e.id
                 AND t2.success
              )::numeric
              /
              COALESCE(tc.cnt, 0)::numeric
            ),
            4
        )
   END AS TSR,
   -- ASR: Attack Success Rate from trajectory JSON (any attack_success = TRUE)
   CASE WHEN COALESCE(tc.cnt, 0) = 0 THEN 0
        ELSE ROUND(
            (
              (SELECT COUNT(*)
               FROM trajectories t3
               WHERE t3.experiment_id = e.id
                 AND attack_success_any(t3.trajectory) IS TRUE
              )::numeric
              /
              COALESCE(tc.cnt, 0)::numeric
            ),
            4
        )
   END AS ASR,
   -- DSR: Defense Success Rate from trajectory JSON (no defense_success = FALSE)
   CASE WHEN COALESCE(tc.cnt, 0) = 0 THEN 0
        ELSE ROUND(
            (
              (SELECT COUNT(*)
               FROM trajectories t4
               WHERE t4.experiment_id = e.id
                 AND defense_success_all(t4.trajectory) IS TRUE
              )::numeric
              /
              COALESCE(tc.cnt, 0)::numeric
            ),
            4
        )
   END AS DSR
FROM experiments e
LEFT JOIN task_datasets td ON td.tasks = e.task_ids
LEFT JOIN traj_counts tc ON tc.experiment_id = e.id
ORDER BY e.id;

-- 2) Create the trigger function
CREATE OR REPLACE FUNCTION experiment_summary_v2_delete()
RETURNS trigger AS $$
BEGIN
  -- actually delete the experiment
  DELETE FROM experiments
   WHERE id = OLD.id;
  RETURN OLD;
END;
$$ LANGUAGE plpgsql;

-- 3) Attach the INSTEAD OF DELETE trigger to the view
DROP TRIGGER IF EXISTS experiment_summary_v2_delete_tr ON experiment_summary_v2;
CREATE TRIGGER experiment_summary_v2_delete_tr
INSTEAD OF DELETE ON experiment_summary_v2
FOR EACH ROW
EXECUTE FUNCTION experiment_summary_v2_delete();

-- 4) Create the trigger function for updates
CREATE OR REPLACE FUNCTION experiment_summary_v2_update()
RETURNS trigger AS $$
BEGIN
  UPDATE experiments
    SET name = NEW.name,
        username = NEW.username,
        description = NEW.description
  WHERE id = OLD.id;
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- 5) Attach the INSTEAD OF UPDATE trigger to the view
DROP TRIGGER IF EXISTS experiment_summary_v2_update_tr ON experiment_summary_v2;
CREATE TRIGGER experiment_summary_v2_update_tr
INSTEAD OF UPDATE ON experiment_summary_v2
FOR EACH ROW
EXECUTE FUNCTION experiment_summary_v2_update();
"""

DEFENSE_EXPERIMENT_SUMMARY_VIEW = """
-- 1) Create the view (or replace if it already exists)
DROP VIEW IF EXISTS defense_experiment_summary_v3 CASCADE;
CREATE OR REPLACE VIEW defense_experiment_summary_v3 AS
WITH step_counts AS (
   SELECT
     defense_experiment_id,
     COUNT(*) AS total_steps,
     SUM(CASE WHEN success THEN 1 ELSE 0 END)::numeric AS successful_steps,
     SUM(CASE WHEN error_message IS NOT NULL THEN 1 ELSE 0 END) AS error_num,
     SUM(CASE WHEN error_message IS NULL AND relevant_cap_set THEN 1 ELSE 0 END)::numeric AS relevant_cap_true_count,
     SUM(CASE WHEN error_message IS NULL AND success AND relevant_cap_set THEN 1 ELSE 0 END)::numeric AS relevant_success_count,
     ROUND(AVG(CASE WHEN error_message IS NULL THEN input_tokens END)::numeric, 2) AS avg_input_tokens,
     ROUND(AVG(CASE WHEN error_message IS NULL THEN output_tokens END)::numeric, 2) AS avg_output_tokens,
     ROUND(AVG(CASE WHEN error_message IS NULL THEN total_request_time END)::numeric, 2) AS avg_request_time,
     ROUND(AVG(CASE WHEN error_message IS NULL THEN (async_messages_stats->>'total_async_requests')::numeric END)::numeric, 2) AS avg_label_requests,
     ROUND(AVG(CASE WHEN error_message IS NULL THEN (async_messages_stats->>'total_input_tokens')::numeric END)::numeric, 2) AS avg_label_input_tokens,
     ROUND(AVG(CASE WHEN error_message IS NULL THEN (async_messages_stats->>'cached_input_tokens')::numeric END)::numeric, 2) AS avg_label_input_tokens_cached,
     ROUND(AVG(CASE WHEN error_message IS NULL THEN (async_messages_stats->>'total_output_tokens')::numeric END)::numeric, 2) AS avg_label_output_tokens,
     ROUND(AVG(CASE WHEN error_message IS NULL THEN (async_messages_stats->>'cached_output_tokens')::numeric END)::numeric, 2) AS avg_label_output_tokens_cached,
     ROUND(AVG(CASE WHEN error_message IS NULL THEN ((async_messages_stats->>'last_request_time')::numeric - (async_messages_stats->>'first_request_time')::numeric) END)::numeric, 2) AS avg_label_duration,
     -- Global cache hit rate across steps: sum(cached_requests) / sum(total_requests)
     ROUND(
       (SUM(CASE WHEN error_message IS NULL THEN async_cached_requests(async_messages_stats) END)::numeric)
       /
       NULLIF(SUM(CASE WHEN error_message IS NULL THEN async_total_requests(async_messages_stats) END)::numeric, 0),
       4
     ) AS global_cache_hit_rate,
     -- Average of per-step cache_hit_rate values
     ROUND(
       AVG(CASE WHEN error_message IS NULL THEN (async_messages_stats->>'cache_hit_rate')::numeric END)::numeric,
       4
     ) AS avg_cache_hit_rate
   FROM defense_harness_steps
   GROUP BY defense_experiment_id
)
SELECT
   dhe.id,
   dhe.name,
   e.name AS reference_experiment_name,
   dhe.username,
   dhe.created_at,
   dhe.description,
   COALESCE(
     dhe.config->'defense'->>'defense_id',
     dhe.config->'defense'->>'type',
     dhe.config->>'defense'
   ) AS defense_id,
   (dhe.config->'defense_kwargs'->'sampling_params'->>'model') AS model,
   COALESCE(dhe.config->'defense_kwargs'->'sampling_params_labeler'->>'model', 'None') AS labeler_model,
   COALESCE(sc.total_steps, 0) AS evaluated_steps,
    ARRAY_LENGTH(dhe.skip_observation_ids, 1) AS skipped_steps,
   COALESCE(sc.error_num, 0) AS error_num,
   CASE 
     WHEN COALESCE(sc.total_steps - sc.error_num, 0) = 0 THEN 0
     ELSE ROUND(sc.successful_steps / (sc.total_steps - sc.error_num), 4)
   END AS UFSR,
   CASE 
     WHEN COALESCE(sc.total_steps - sc.error_num, 0) = 0 THEN 0
     ELSE ROUND(COALESCE(sc.relevant_cap_true_count, 0) / (sc.total_steps - sc.error_num), 4)
   END AS CSR,
   CASE 
     WHEN COALESCE(sc.total_steps - sc.error_num, 0) = 0 THEN 0
     ELSE ROUND(COALESCE(sc.relevant_success_count, 0) / (sc.total_steps - sc.error_num), 4)
   END AS UFCSR,
   COALESCE(sc.avg_input_tokens, 0) AS avg_input_tokens,
   COALESCE(sc.avg_output_tokens, 0) AS avg_output_tokens,
   COALESCE(sc.avg_request_time, 0) AS avg_request_time,
   COALESCE(sc.avg_label_requests, 0) AS avg_label_requests,
   COALESCE(sc.avg_label_input_tokens, 0) AS avg_label_input_tokens,
   COALESCE(sc.avg_label_input_tokens_cached, 0) AS avg_label_input_tokens_cached,
   COALESCE(sc.avg_label_output_tokens, 0) AS avg_label_output_tokens,
   COALESCE(sc.avg_label_output_tokens_cached, 0) AS avg_label_output_tokens_cached,
   COALESCE(sc.avg_label_duration, 0) AS avg_label_duration,
   COALESCE(sc.avg_cache_hit_rate, 0) AS avg_cache_hit_rate,
   COALESCE(sc.global_cache_hit_rate, 0) AS global_cache_hit_rate
FROM defense_harness_experiments dhe
LEFT JOIN experiments e ON e.id = dhe.reference_experiment_id
LEFT JOIN step_counts sc ON sc.defense_experiment_id = dhe.id
ORDER BY dhe.id;

-- 2) Create the trigger function
CREATE OR REPLACE FUNCTION defense_experiment_summary_v3_delete()
RETURNS trigger AS $$
BEGIN
  -- actually delete the defense experiment
  DELETE FROM defense_harness_experiments
   WHERE id = OLD.id;
  RETURN OLD;
END;
$$ LANGUAGE plpgsql;

-- 3) Attach the INSTEAD OF DELETE trigger to the view
DROP TRIGGER IF EXISTS defense_experiment_summary_v3_delete_tr ON defense_experiment_summary_v3;
CREATE TRIGGER defense_experiment_summary_v3_delete_tr
INSTEAD OF DELETE ON defense_experiment_summary_v3
FOR EACH ROW
EXECUTE FUNCTION defense_experiment_summary_v3_delete();

-- 4) Create the trigger function for updates
CREATE OR REPLACE FUNCTION defense_experiment_summary_v3_update()
RETURNS trigger AS $$
BEGIN
  UPDATE defense_harness_experiments
    SET name = NEW.name,
        username = NEW.username,
        description = NEW.description
  WHERE id = OLD.id;
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- 5) Attach the INSTEAD OF UPDATE trigger to the view
DROP TRIGGER IF EXISTS defense_experiment_summary_v3_update_tr ON defense_experiment_summary_v3;
CREATE TRIGGER defense_experiment_summary_v3_update_tr
INSTEAD OF UPDATE ON defense_experiment_summary_v3
FOR EACH ROW
EXECUTE FUNCTION defense_experiment_summary_v3_update();
"""

ALL_VIEWS = [EXPERIMENT_SUMMARY_VIEW, DEFENSE_EXPERIMENT_SUMMARY_VIEW]

if __name__ == "__main__":
    for view in ALL_VIEWS:
        print(view)
