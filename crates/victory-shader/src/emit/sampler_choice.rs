//! Sampling through a uniform choice of immediate sampler descriptors.

use super::{Inst, Lower, Source, Val, Why};

impl<'a> Lower<'a> {
    pub(super) fn image(&mut self, inst: &Inst, b: usize, i: usize) -> Result<(), Why> {
        let sampler = self.descriptors.get(&((b, i), true)).copied();
        if let Some(Source::Phi { block, args }) = sampler
            && let [first @ Source::Constant(_), second @ Source::Constant(_)] = args.as_slice()
            && let [left, right] = self.x.p.blocks[*block].preds.as_slice()
        {
            // Immediate samplers are separate static bindings. Sample both with
            // the original derivatives, then select the texel on the same edge
            // that supplied the console's sampler.
            if self.block[*left].is_none() {
                return self.image_with_sampler(inst, b, i, Some(second));
            }
            if self.block[*right].is_none() {
                return self.image_with_sampler(inst, b, i, Some(first));
            }
            let edge = |pred: usize| {
                self.edges.get(&(pred, *block)).copied().or(self.tail[pred])
                    .ok_or("a sampler choice with an unfinished predecessor")
            };
            let (left, right) = (edge(*left)?, edge(*right)?);
            let join = self.block[*block].ok_or("a sampler choice without a host block")?;
            let chosen = self.s.m.phi(join, self.s.t.i1);
            let (yes, no) = (self.s.m.bool(true), self.s.m.bool(false));
            self.s.m.set_phi(chosen, vec![(yes, left), (no, right)]);
            self.image_with_sampler(inst, b, i, Some(first))?;
            let left = inst.writes.iter()
                .map(|&(_, v)| self.i32_of(v))
                .collect::<Result<Vec<_>, _>>()?;
            self.image_with_sampler(inst, b, i, Some(second))?;
            for (&(_, v), a) in inst.writes.iter().zip(left) {
                let b = self.i32_of(v)?;
                let result = self.s.m.select(self.cur, chosen, a, b);
                self.vals.insert(v, Val::I32(result));
            }
            return Ok(());
        }
        self.image_with_sampler(inst, b, i, sampler)
    }
}
