use baselines::{Method, Method2D, baseline, baseline_2d_with, baseline_with, correct};

#[test]
fn root_level_simple_api_is_usable_without_builders() {
    let y: Vec<f64> = (0..32)
        .map(|index| 1.0 + index as f64 * 0.02 + if index == 15 { 3.0 } else { 0.0 })
        .collect();

    assert_eq!(baseline(&y).unwrap().len(), y.len());
    assert_eq!(correct(&y).unwrap().len(), y.len());
    assert_eq!(baseline_with(&y, Method::Arpls).unwrap().len(), y.len());
}

#[test]
fn root_level_two_dimensional_simple_api_preserves_shape() {
    let data = vec![2.0; 5 * 6];
    let fitted = baseline_2d_with(&data, 5, 6, Method2D::RollingBall).unwrap();
    assert_eq!(fitted.len(), data.len());
}
