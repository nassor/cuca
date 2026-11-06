//! Stop-string matching over a streamed text.
//!
//! [`StopMatcher`] sits between the output parser and the stream. Text arrives
//! in arbitrary fragments, so a stop string can be split across pushes: the
//! matcher holds back the shortest tail that could still grow into a stop
//! string (never more than `longest stop - 1` bytes, always cut on a UTF-8
//! character boundary) and releases everything before it. With no stop string
//! it buffers nothing.

/// What one [`StopMatcher::push`] releases.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct StopOutcome {
    /// Text safe to emit now: it cannot be part of a stop string.
    pub(crate) emit: String,
    /// A stop string occurred: `emit` ends right before it and generation
    /// ends.
    pub(crate) done: bool,
}

/// Cuts a text stream before the first occurrence of any stop string.
#[derive(Debug)]
pub(crate) struct StopMatcher {
    /// Non-empty stop strings.
    stops: Vec<String>,
    /// Longest stop string, in bytes.
    longest: usize,
    /// Text received but not yet released: a tail that may start a stop
    /// string.
    held: String,
}

impl StopMatcher {
    /// A matcher for `stop`; empty entries are ignored (they would match
    /// everywhere).
    pub(crate) fn new(stop: &[String]) -> Self {
        let stops: Vec<String> = stop.iter().filter(|s| !s.is_empty()).cloned().collect();
        let longest = stops.iter().map(String::len).max().unwrap_or(0);
        Self {
            stops,
            longest,
            held: String::new(),
        }
    }

    /// Whether there is no stop string: every push passes straight through.
    pub(crate) fn is_passthrough(&self) -> bool {
        self.stops.is_empty()
    }

    /// Append `text`; release the part that no stop string can still reach,
    /// or, when a stop string is complete, the text before its first
    /// occurrence with `done` set.
    pub(crate) fn push(&mut self, text: &str) -> StopOutcome {
        if self.stops.is_empty() {
            return StopOutcome {
                emit: text.to_string(),
                done: false,
            };
        }
        self.held.push_str(text);
        let earliest = self
            .stops
            .iter()
            .filter_map(|s| self.held.find(s.as_str()))
            .min();
        if let Some(at) = earliest {
            self.held.truncate(at);
            return StopOutcome {
                emit: std::mem::take(&mut self.held),
                done: true,
            };
        }
        // No stop string is complete: keep the longest tail that is a proper
        // prefix of one (a full match was ruled out above).
        let max_keep = self.held.len().min(self.longest - 1);
        let keep = (1..=max_keep)
            .rev()
            .find(|&n| {
                let from = self.held.len() - n;
                self.held.is_char_boundary(from) && {
                    let tail = &self.held[from..];
                    self.stops.iter().any(|s| s.starts_with(tail))
                }
            })
            .unwrap_or(0);
        let release = self.held.len() - keep;
        let emit = self.held[..release].to_string();
        self.held.drain(..release);
        StopOutcome { emit, done: false }
    }

    /// Release the held text: generation ended without completing a stop
    /// string.
    pub(crate) fn flush(&mut self) -> String {
        std::mem::take(&mut self.held)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn matcher(stop: &[&str]) -> StopMatcher {
        StopMatcher::new(&stop.iter().map(|s| s.to_string()).collect::<Vec<_>>())
    }

    #[test]
    fn a_stop_string_split_across_pushes_is_cut() {
        let mut m = matcher(&["bc"]);
        let first = m.push("ab");
        assert_eq!(
            first,
            StopOutcome {
                emit: "a".into(),
                done: false
            }
        );
        let second = m.push("c");
        assert_eq!(
            second,
            StopOutcome {
                emit: String::new(),
                done: true
            }
        );
        assert_eq!(m.flush(), "");
    }

    #[test]
    fn a_stop_string_inside_one_push_cuts_before_it() {
        let mut m = matcher(&["END"]);
        let out = m.push("hello END tail");
        assert_eq!(out.emit, "hello ");
        assert!(out.done);
    }

    #[test]
    fn the_earliest_occurrence_of_any_stop_wins() {
        let mut m = matcher(&["zz", "b"]);
        let out = m.push("abzz");
        assert_eq!(out.emit, "a");
        assert!(out.done);
    }

    #[test]
    fn no_match_releases_text_and_flush_returns_the_held_tail() {
        let mut m = matcher(&["xyz"]);
        let first = m.push("abx");
        assert_eq!(first.emit, "ab");
        assert!(!first.done);
        // A tail that stops being a prefix is released with the next text.
        let second = m.push("q");
        assert_eq!(second.emit, "xq");
        assert!(!second.done);
        let third = m.push("xy");
        assert_eq!(third.emit, "");
        assert_eq!(m.flush(), "xy");
        assert_eq!(m.flush(), "");
    }

    #[test]
    fn the_held_tail_never_exceeds_the_longest_stop_minus_one() {
        let mut m = matcher(&["abc"]);
        let mut emitted = String::new();
        for chunk in ["aa", "aa", "ab", "ab"] {
            emitted.push_str(&m.push(chunk).emit);
            assert!(m.held.len() < 3, "held {:?}", m.held);
        }
        emitted.push_str(&m.flush());
        assert_eq!(emitted, "aaaaabab");
    }

    #[test]
    fn multi_byte_characters_are_held_on_a_boundary() {
        // The stop string starts with a 3-byte character.
        let mut m = matcher(&["€uro"]);
        let first = m.push("pay €");
        assert_eq!(first.emit, "pay ");
        assert!(!first.done);
        let second = m.push("uro now");
        assert_eq!(second.emit, "");
        assert!(second.done);

        // A tail inside a character is never split.
        let mut m = matcher(&["éa"]);
        let out = m.push("xé");
        assert_eq!(out.emit, "x");
        assert_eq!(m.flush(), "é");
        let mut m = matcher(&["ab"]);
        let out = m.push("日本a");
        assert_eq!(out.emit, "日本");
        assert_eq!(m.flush(), "a");
    }

    #[test]
    fn an_empty_stop_list_passes_through_without_buffering() {
        let mut m = matcher(&[]);
        assert!(m.is_passthrough());
        let out = m.push("anything");
        assert_eq!(out.emit, "anything");
        assert!(!out.done);
        assert_eq!(m.flush(), "");
        // Empty entries are ignored too.
        assert!(matcher(&[""]).is_passthrough());
    }
}
