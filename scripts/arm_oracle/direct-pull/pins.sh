# Tag commits verified against github.com/google-deepmind/mujoco tags and doc/changelog.rst.
# Tree SHAs are git rev-parse COMMIT^{tree} of those commits.
# Dependency SHAs are the checkouts CMake fetched while building the x86 contract-off oracle.

menagerie_commit() { echo 4d038b3feae26ec82b46a4d586379114012a8ac7; }

mujoco_commit() {
  case "$1" in
    3.14.0) echo 9ecbb9d7b5ee623f54745638d36799ff90e6f7cd ;;
    3.15.0) echo 9ea3cdfcae93bf2cc4dc0e1a1627c5a39a1e06e5 ;;
    3.16.0) echo a892750516336d8358949f4518fcc4fbbd85beae ;;
    3.17.0) echo 800a109a4bdd02cb60404d565db4ae16fb04a53b ;;
    4.0.0) echo ac60eae30fa7e0d31315c38d9476752fb81be67e ;;
    *) return 1 ;;
  esac
}

mujoco_tree() {
  case "$1" in
    3.14.0) echo cece271bb60f5b3b07273f98ce71a6675b76444b ;;
    3.15.0) echo 9dc96300a8b94fccb96f94c3c79aca6edcb0d80a ;;
    3.16.0) echo ad5b62f51a8ad9ed87439de2a87bbc51ba5f9159 ;;
    3.17.0) echo dea445731888479258ad1c34dcfbcab9ef75a832 ;;
    4.0.0) echo 6008e06cb48f7836818364cb808fe033e72d88e6 ;;
    *) return 1 ;;
  esac
}

# name version -> commit. Versions that share a pin repeat it.
dep_commit() {
  local name="$1" ver="$2"
  case "$name:$ver" in
    ccd:*) echo 7931e764a19ef6b21b443376c699bbc9c6d4fba8 ;;
    qhull:*) echo d1c2fc0caa5f644f3a0f220290d4a868c68ed4f6 ;;
    tinyobjloader:*) echo 2945a967c5303b2c8c14174117c45f3302591150 ;;
    tinyxml2:*) echo e6caeae85799003f4ca74ff26ee16a789bc2af48 ;;
    marchingcubecpp:*) echo f03a1b3ec29b1d7d865691ca8aea4f1eb2c2873d ;;
    lodepng:3.14.0) echo 17d08dd26cac4d63f43af217ebd70318bfb8189c ;;
    lodepng:*) echo 22561883dd63fd1850f18e1f6adac321e4f609b0 ;;
    miniz:3.14.0|miniz:3.15.0) echo d10b03cc73475af673df40f06e5cefd1d5f940d9 ;;
    miniz:*) echo 77d0dce8627735138c51770d1799a1ef48f2117d ;;
    eigen3:3.14.0) echo ea13a98decd497a8c5588fb5de71b57bcf10d864 ;;
    eigen3:3.15.0) echo 087757ad50159b5bd86ab73e4df34e6b4ea74258 ;;
    eigen3:3.16.0|eigen3:3.17.0) echo b1f8b0c3b8db9c3711349144acf951226700ddb4 ;;
    eigen3:4.0.0) echo 1da510b8155f8fa81924616dcc517ecc4202e4d3 ;;
    *) return 1 ;;
  esac
}
