use std::collections::{BTreeMap, BTreeSet};

const BENCHMARKS: &str = include_str!("../benches/baseline_workloads.rs");

#[test]
fn criterion_workloads_cover_public_algorithm_functions() {
    let groups = benchmark_groups();

    assert_public_functions_are_benchmarked(
        "whittaker_1d",
        &[
            include_str!("../src/whittaker/asls.rs"),
            include_str!("../src/whittaker/airpls.rs"),
            include_str!("../src/whittaker/arpls.rs"),
            include_str!("../src/whittaker/variants.rs"),
        ],
        "whittaker_1d",
        Some("whittaker_1d_into"),
        "_256",
        &[],
        &groups,
    );
    assert_public_functions_are_benchmarked(
        "polynomial_1d",
        &[include_str!("../src/polynomial/mod.rs")],
        "polynomial_1d",
        Some("polynomial_1d_into"),
        "_256",
        &[],
        &groups,
    );
    assert_public_functions_are_benchmarked(
        "morphology_1d",
        &[include_str!("../src/morphology/mod.rs")],
        "morphology_1d",
        Some("morphology_1d_into"),
        "_256",
        &[],
        &groups,
    );
    assert_public_functions_are_benchmarked(
        "smoothing_1d",
        &[include_str!("../src/smoothing/mod.rs")],
        "smoothing_1d",
        Some("smoothing_1d_into"),
        "_256",
        &[],
        &groups,
    );
    assert_public_functions_are_benchmarked(
        "spline_1d",
        &[
            include_str!("../src/spline/mod.rs"),
            include_str!("../src/spline/corner.rs"),
            include_str!("../src/spline/mixture.rs"),
        ],
        "spline_1d",
        None,
        "_256",
        &[],
        &groups,
    );
    assert_public_functions_are_benchmarked(
        "classification_1d",
        &[include_str!("../src/classification/mod.rs")],
        "classification_1d",
        None,
        "_256",
        &[],
        &groups,
    );
    assert_public_functions_are_benchmarked(
        "optimizers_misc_1d",
        &[
            include_str!("../src/optimizers/mod.rs"),
            include_str!("../src/misc/mod.rs"),
            include_str!("../src/misc/beads.rs"),
        ],
        "optimizers_misc_1d",
        None,
        "_256",
        &[("collab_pls", "collab_pls_3x256")],
        &groups,
    );

    assert_public_functions_are_benchmarked(
        "whittaker_2d",
        &[
            include_str!("../src/two_d/whittaker.rs"),
            include_str!("../src/two_d/whittaker_eigen.rs"),
        ],
        "whittaker_2d",
        Some("whittaker_2d_into"),
        "_16x16",
        &[],
        &groups,
    );
    assert_public_functions_are_benchmarked(
        "polynomial_2d",
        &[include_str!("../src/two_d/polynomial.rs")],
        "polynomial_2d",
        Some("polynomial_2d_into"),
        "_16x16",
        &[],
        &groups,
    );
    assert_public_functions_are_benchmarked(
        "morphology_2d",
        &[include_str!("../src/two_d/morphology.rs")],
        "morphology_2d",
        Some("morphology_2d_into"),
        "_16x16",
        &[],
        &groups,
    );
    assert_public_functions_are_benchmarked(
        "spline_2d",
        &[include_str!("../src/two_d/spline.rs")],
        "spline_2d",
        Some("spline_2d_into"),
        "_16x16",
        &[],
        &groups,
    );
    assert_public_functions_are_benchmarked(
        "optimizers_2d",
        &[include_str!("../src/two_d/optimizers.rs")],
        "optimizers_2d",
        Some("optimizers_2d_into"),
        "_16x16",
        &[("collab_pls", "collab_pls_2x16x16")],
        &groups,
    );
    assert_public_functions_are_benchmarked(
        "batch_cpu",
        &[include_str!("../src/backend/cpu.rs")],
        "batch_cpu",
        None,
        "",
        &[("snip_batch_into", "snip_batch_cpu_16x256")],
        &groups,
    );
}

fn assert_public_functions_are_benchmarked(
    family: &str,
    sources: &[&str],
    allocating_group: &str,
    into_group: Option<&str>,
    suffix: &str,
    overrides: &[(&str, &str)],
    groups: &BTreeMap<String, BTreeSet<String>>,
) {
    let overrides = overrides.iter().copied().collect::<BTreeMap<_, _>>();

    for function in public_functions(sources) {
        let benchmark_name = overrides
            .get(function.as_str())
            .map_or_else(|| format!("{function}{suffix}"), |name| (*name).to_owned());
        let group = if is_into_api(&function) && !overrides.contains_key(function.as_str()) {
            into_group.unwrap_or_else(|| panic!("{family}/{function} has no into benchmark group"))
        } else {
            allocating_group
        };

        assert!(
            groups
                .get(group)
                .is_some_and(|names| names.contains(&benchmark_name)),
            "missing Criterion benchmark for public function {family}::{function}; \
             expected {group}/{benchmark_name}"
        );
    }
}

fn is_into_api(function: &str) -> bool {
    function.ends_with("_into") || function.ends_with("_into_with_history")
}

fn public_functions(sources: &[&str]) -> BTreeSet<String> {
    let mut functions = BTreeSet::new();
    for source in sources {
        for line in source.lines() {
            if let Some(rest) = line.strip_prefix("pub fn ") {
                let end = rest
                    .find(['(', '<'])
                    .expect("public function declaration has a name terminator");
                functions.insert(rest[..end].to_owned());
            }
        }
    }
    functions
}

fn benchmark_groups() -> BTreeMap<String, BTreeSet<String>> {
    let mut groups = BTreeMap::<String, BTreeSet<String>>::new();
    let mut current_group = None::<String>;
    let mut pending_macro_name = false;

    for line in BENCHMARKS.lines() {
        if let Some(group) = quoted_after(line, "benchmark_group(") {
            current_group = Some(group);
            continue;
        }

        if pending_macro_name {
            if let Some(name) = first_quoted(line) {
                insert_benchmark(&mut groups, &current_group, name);
                pending_macro_name = false;
            }
            continue;
        }

        if let Some(name) = quoted_after(line, "bench_function(") {
            insert_benchmark(&mut groups, &current_group, name);
        } else if line.contains("bench_1d_into!(") || line.contains("bench_2d_into!(") {
            if let Some(name) = first_quoted(line) {
                insert_benchmark(&mut groups, &current_group, name);
            } else {
                pending_macro_name = true;
            }
        }
    }

    groups
}

fn insert_benchmark(
    groups: &mut BTreeMap<String, BTreeSet<String>>,
    current_group: &Option<String>,
    name: String,
) {
    let group = current_group
        .as_ref()
        .expect("benchmark name appeared before benchmark_group");
    groups.entry(group.clone()).or_default().insert(name);
}

fn quoted_after(line: &str, marker: &str) -> Option<String> {
    let rest = line.split_once(marker)?.1;
    first_quoted(rest)
}

fn first_quoted(line: &str) -> Option<String> {
    let start = line.find('"')? + 1;
    let end = line[start..].find('"')?;
    Some(line[start..start + end].to_owned())
}
