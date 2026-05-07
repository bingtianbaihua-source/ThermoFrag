# vendor/

Code copied from external repositories so this project is self-contained. Each subdir is unmodified upstream code unless noted.

## bbar_fragmentation/

BRICS fragmentation utilities from `BBAR` (Seo et al., Adv Sci 2023). Used to build the fragment library from ZINC. Original at https://github.com/SeonghwanSeo/BBAR

## bbar_transform/

Atom and bond featurizers and the molecule-to-graph transform from BBAR. Used as a reference implementation; ThermoFrag's own featurizer in `src/thermofrag/data/` may diverge.

## bbar_utils/

Common utility helpers from BBAR (typing aliases and IO).

## License

These files are redistributed under the original BBAR license (MIT). See the upstream repository for full terms.
