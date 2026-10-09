source deployment/local-env.sh
cd RoboSynChallenge

bash policy/pi05_v2/eval.sh click_bell random \
  /path/to/checkpoint \
  4 --max_episodes 20 --headless true