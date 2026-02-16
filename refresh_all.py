import sync_manual_ranking
import score_normalize
import visual_audit_ranking
import time

def run_master_workflow():
    start_time = time.time()
    print("🚀 [1/3] 開始同步資料夾內的手動修改分數...")
    try:
        sync_manual_ranking.sync_scores_from_filenames()
        print("✅ 同步完成。")
    except Exception as e:
        print(f"❌ 同步失敗: {e}")
        return

    print("\n🚀 [2/3] 開始執行分數標準化 (Score Normalization)...")
    try:
        score_normalize.normalize_dataset()
        print("✅ 標準化完成。")
    except Exception as e:
        print(f"❌ 標準化失敗: {e}")
        return

    print("\n🚀 [3/3] 重新生成排行榜資料夾 (Visual Audit)...")
    try:
        visual_audit_ranking.run_visual_audit()
        print("✅ 排行榜已更新。")
    except Exception as e:
        print(f"❌ 重新生成失敗: {e}")
        return

    end_time = time.time()
    print(f"\n✨ 整個流程執行完畢！總耗時: {end_time - start_time:.2f} 秒")
    print("👉 現在你可以回去 ./static/for_ranking/ 查看最新的視覺排序了。")

if __name__ == "__main__":
    run_master_workflow()