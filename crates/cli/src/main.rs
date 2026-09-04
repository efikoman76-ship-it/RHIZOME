//! The `rhizome` command line entry point.
//!
//! Implemented subcommands in this milestone: `params`, `inspect`,
//! `verify` and `spec-check`. Each is a thin shell over the library crates
//! so that behaviour is covered by library unit tests as well.

#![forbid(unsafe_code)]

use rhizome_core::params::{forward_flops, markdown_table, report};
use rhizome_core::{ModelConfig, ParamReport};
use rhizome_plan::graph::{Graph, OpKind, Phase};
use rhizome_plan::{arena, plan_hash};
use std::process::ExitCode;

/// Configuration files shipped in `configs/`, in reporting order.
const CONFIG_FILES: [&str; 6] = [
    "configs/test-s.toml",
    "configs/test-m.toml",
    "configs/test-l.toml",
    "configs/edge.toml",
    "configs/standard.toml",
    "configs/flagship.toml",
];

fn load_all() -> Result<Vec<ModelConfig>, String> {
    let mut out = Vec::new();
    for path in CONFIG_FILES {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
        let cfg = ModelConfig::from_toml(&text).map_err(|e| format!("{path}: {e}"))?;
        out.push(cfg);
    }
    Ok(out)
}

fn render(reports: &[ParamReport], format: &str) -> String {
    match format {
        "markdown" => markdown_table(reports),
        _ => {
            let mut s = String::new();
            for r in reports {
                s.push_str(&format!(
                    "{}: total={} active_l1={} per_iteration={} deploy_bytes={}\n",
                    r.name, r.total, r.active_l1, r.per_iteration, r.deploy_bytes
                ));
            }
            s
        }
    }
}

fn cmd_params(args: &[String]) -> Result<String, String> {
    let format = args
        .iter()
        .find_map(|a| a.strip_prefix("--format="))
        .unwrap_or("text");
    let cfgs = load_all()?;
    let reports: Vec<ParamReport> = cfgs.iter().map(report).collect();
    Ok(render(&reports, format))
}

fn cmd_inspect(args: &[String]) -> Result<String, String> {
    let path = args
        .iter()
        .find(|a| !a.starts_with("--"))
        .ok_or("usage: rhizome inspect <config.toml>")?;
    let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
    let cfg = ModelConfig::from_toml(&text).map_err(|e| e.to_string())?;
    let r = report(&cfg);
    Ok(format!(
        "name={}\nd_model={}\ntrunk_blocks={}\nl_max={}\n\
front_end={}\ntotal_params={}\nflops_l1={}\nflops_lmax={}\n",
        cfg.name,
        cfg.d_model,
        cfg.trunk_blocks(),
        cfg.l_max,
        cfg.front_end.name(),
        r.total,
        forward_flops(&cfg, 1),
        forward_flops(&cfg, cfg.l_max)
    ))
}

fn smoke_graph() -> Result<Graph, String> {
    let mut g = Graph::new();
    let x = g.tensor("x", 256, rhizome_core::DType::F32, false);
    let w = g.tensor("w", 256 * 256, rhizome_core::DType::BF16, true);
    let h = g
        .push(OpKind::Linear, &[x, w], "h", 256, rhizome_core::DType::F32)
        .map_err(|e| e.to_string())?;
    let a = g
        .push(OpKind::Silu, &[h], "a", 256, rhizome_core::DType::F32)
        .map_err(|e| e.to_string())?;
    g.push(
        OpKind::ResidualAdd,
        &[a, x],
        "y",
        256,
        rhizome_core::DType::F32,
    )
    .map_err(|e| e.to_string())?;
    Ok(g)
}

fn cmd_verify() -> Result<String, String> {
    let cfgs = load_all()?;
    for c in &cfgs {
        c.validate().map_err(|e| format!("{}: {e}", c.name))?;
    }
    let g = smoke_graph()?;
    g.validate().map_err(|e| e.to_string())?;
    let b = g.backward().map_err(|e| e.to_string())?;
    let p = arena::plan(&b).map_err(|e| e.to_string())?;
    p.verify().map_err(|e| e.to_string())?;
    Ok(format!(
        "verify: {} configs ok, backward nodes={}, arena bytes={}, plan hash={:016x}\n",
        cfgs.len(),
        b.nodes().len(),
        p.total_bytes,
        plan_hash(&b, Phase::Train, 1)
    ))
}

fn cmd_spec_check() -> Result<String, String> {
    let cfgs = load_all()?;
    let reports: Vec<ParamReport> = cfgs.iter().map(report).collect();
    let table = markdown_table(&reports);
    let spec = std::fs::read_to_string("SPEC.md").map_err(|e| format!("SPEC.md: {e}"))?;
    let start = spec
        .find("<!-- BEGIN PARAMS TABLE -->")
        .ok_or("SPEC.md is missing the params table markers")?;
    let end = spec
        .find("<!-- END PARAMS TABLE -->")
        .ok_or("SPEC.md is missing the params table end marker")?;
    let embedded = spec[start..end]
        .lines()
        .skip(1)
        .collect::<Vec<_>>()
        .join("\n");
    if embedded.trim() == table.trim() {
        Ok("spec-check: SPEC.md params table matches `rhizome params`\n".to_string())
    } else {
        Err(format!(
            "spec-check: SPEC.md params table is stale.\n\
--- expected ---\n{table}\n--- found ---\n{embedded}\n"
        ))
    }
}

fn usage() -> String {
    "rhizome <command>\n\ncommands:\n\
  params [--format=markdown|text]\n  inspect <config.toml>\n\
  verify\n  spec-check\n"
        .to_string()
}

fn run() -> Result<String, String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(cmd) = args.first() else {
        return Ok(usage());
    };
    let rest = &args[1..];
    match cmd.as_str() {
        "params" => cmd_params(rest),
        "inspect" => cmd_inspect(rest),
        "verify" => cmd_verify(),
        "spec-check" => cmd_spec_check(),
        "--help" | "-h" | "help" => Ok(usage()),
        other => Err(format!("unknown command `{other}`\n\n{}", usage())),
    }
}

fn main() -> ExitCode {
    match run() {
        Ok(out) => {
            print!("{out}");
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}
