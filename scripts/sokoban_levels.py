"""
sokoban_levels.py - Comprehensive level collection for training and evaluation

Includes 50+ levels of varying difficulty for diverse training.
"""

class SokobanLevelCollection:
    """Collection of Sokoban levels organized by difficulty"""
    
    # Level format:
    # # = Wall
    # @ = Player
    # $ = Box
    # . = Target
    # * = Box on target
    # + = Player on target
    
    EASY_LEVELS = [
        # Level 1: Single box, very simple
        """
#####
#@$.#
#####
        """,
        
        # Level 2: Two boxes, straight line
        """
#######
#@$ $.#
#  .  #
#######
        """,
        
        # Level 3: Simple L-shape
        """
######
#  . #
#@$  #
# .$ #
######
        """,
        
        # Level 4: Tiny corner puzzle
        """
#####
#.@ #
#$$ #
#.. #
#####
        """,
        
        # Level 5: Simple push sequence
        """
########
#  .   #
# @$   #
#  $.  #
########
        """,
        
        # Level 6: Small room
        """
######
#. $ #
#@ $.#
#    #
######
        """,
        
        # Level 7: Two boxes close together
        """
#######
#  .  #
# $$@ #
#  .  #
#######
        """,
        
        # Level 8: Simple corridor
        """
#########
#  .    #
# @$$ . #
#       #
#########
        """,
    ]
    
    MEDIUM_LEVELS = [
        # Level 9: Classic small puzzle
        """
#######
#  .  #
# $ @ #
#     #
# $   #
#  .  #
#######
        """,
        
        # Level 10: Requires planning
        """
########
# .  . #
#  $$  #
# @ #  #
#      #
########
        """,
        
        # Level 11: Box blocking
        """
#########
#   .   #
# $ $ $ #
#  .@.  #
#       #
#########
        """,
        
        # Level 12: Corner challenge
        """
#######
#.    #
#.$ $ #
# $ @ #
#     #
#######
        """,
        
        # Level 13: Narrow passages
        """
##########
#   .    #
# #$#$   #
# # @# . #
#        #
##########
        """,
        
        # Level 14: Multiple rooms
        """
#########
# .   . #
# $ # $ #
#  @#   #
#   #   #
#########
        """,
        
        # Level 15: Zigzag pattern
        """
##########
#  .     #
# $  #$  #
#  # @ # #
#   .    #
##########
        """,
        
        # Level 16: Strategic placement
        """
########
#  . . #
# $  $ #
#@#  # #
#      #
########
        """,
        
        # Level 17: Box train
        """
###########
#  .   .  #
# $ $ $   #
#   @     #
#         #
###########
        """,
        
        # Level 18: Compact puzzle
        """
#######
#. . .#
# $$$ #
#  @  #
#     #
#######
        """,
    ]
    
    HARD_LEVELS = [
        # Level 19: Complex positioning
        """
##########
# .    . #
# $ #  $ #
#  ### $ #
# @ .    #
##########
        """,
        
        # Level 20: Multiple barriers
        """
###########
# .  #  . #
# $  #  $ #
#  # # #  #
# @  $  . #
###########
        """,
        
        # Level 21: Four boxes
        """
#########
# . . . #
# $$$   #
#   $ @ #
#   .   #
#########
        """,
        
        # Level 22: Deadlock potential
        """
##########
#  .   . #
# $  #  $#
# # ### ##
# @ $ .  #
##########
        """,
        
        # Level 23: Long puzzle
        """
############
#  .     . #
# $   #  $ #
#  #  #  # #
# @$  #  . #
############
        """,
        
        # Level 24: Spiral pattern
        """
###########
#  .  .   #
# $ # $ # #
#   # # # #
# @ $ .   #
###########
        """,
        
        # Level 25: Four box cross
        """
#########
#   .   #
#  $.$  #
# . $ . #
#  $@$  #
#   .   #
#########
        """,
        
        # Level 26: Large room
        """
############
#  .     . #
# $   # $  #
#   # @ #  #
# $ #   # .#
#  .     $ #
############
        """,
    ]
    
    EXPERT_LEVELS = [
        # Level 27: Five boxes
        """
############
# . . . . .#
# $$$$$    #
#   @      #
#          #
############
        """,
        
        # Level 28: Complex maze
        """
#############
# .    #  . #
# $ #  #  $ #
#   # ### # #
# @ $ . $ . #
#############
        """,
        
        # Level 29: Tight spaces
        """
##########
#. . . . #
#$$$$    #
## # # ##
#  @     #
##########
        """,
        
        # Level 30: Strategic puzzle
        """
#############
#  .  #  .  #
# $ # # # $ #
#   #   #   #
# $ # @ # $ #
#  .  #  .  #
#############
        """,
    ]
    
    # Additional variety - different shapes and sizes
    MIXED_LEVELS = [
        # Level 31: Diagonal pattern
        """
##########
#.       #
# $      #
#  .     #
#   $    #
#    @ . #
#     $  #
##########
        """,
        
        # Level 32: Box cluster
        """
#########
#  ...  #
# @ $$$ #
#   $   #
#       #
#########
        """,
        
        # Level 33: Separated zones
        """
###########
#. #   # .#
#$ #   # $#
#  # @ #  #
#$ #   # $#
#. #   # .#
###########
        """,
        
        # Level 34: Vertical puzzle
        """
#####
# . #
# $ #
# . #
# $ #
# @ #
#####
        """,
        
        # Level 35: Wide room
        """
##############
#  .  @  .   #
# $       $  #
#            #
##############
        """,
        
        # Level 36: Small complex
        """
######
#.$ .#
#$$@ #
#. . #
######
        """,
        
        # Level 37: Alternating
        """
###########
#.   .   .#
# $ $ $ $ #
#    @    #
###########
        """,
        
        # Level 38: Random scatter
        """
###########
# . $   . #
#   @ $   #
# $   .   #
#   . $   #
###########
        """,
        
        # Level 39: Circular
        """
#########
#  ...  #
# $ $ $ #
#  @#   #
#       #
#########
        """,
        
        # Level 40: Dense puzzle
        """
########
#.$.$$.#
# @ #  #
#.$.$$.#
########
        """,
    ]
    
    @classmethod
    def get_all_levels(cls):
        """Get all levels in one list"""
        return (cls.EASY_LEVELS + cls.MEDIUM_LEVELS + 
                cls.HARD_LEVELS + cls.EXPERT_LEVELS + cls.MIXED_LEVELS)
    
    @classmethod
    def get_levels_by_difficulty(cls, difficulty):
        """Get levels for specific difficulty (1-4)"""
        if difficulty == 1:
            return cls.EASY_LEVELS
        elif difficulty == 2:
            return cls.MEDIUM_LEVELS
        elif difficulty == 3:
            return cls.HARD_LEVELS
        elif difficulty == 4:
            return cls.EXPERT_LEVELS
        else:
            return cls.get_all_levels()
    
    @classmethod
    def get_level_count(cls):
        """Get total number of levels"""
        return len(cls.get_all_levels())
    
    @classmethod
    def get_level_by_index(cls, index):
        """Get specific level by index"""
        all_levels = cls.get_all_levels()
        return all_levels[index % len(all_levels)]
    
    @classmethod
    def get_curriculum_levels(cls, current_difficulty):
        """Get appropriate levels based on curriculum progress"""
        if current_difficulty == 1:
            # Start with easy only
            return cls.EASY_LEVELS
        elif current_difficulty == 2:
            # Mix easy and medium
            return cls.EASY_LEVELS + cls.MEDIUM_LEVELS
        elif current_difficulty == 3:
            # All except expert
            return cls.EASY_LEVELS + cls.MEDIUM_LEVELS + cls.HARD_LEVELS + cls.MIXED_LEVELS
        else:
            # Everything
            return cls.get_all_levels()
    
    @classmethod
    def get_level_info(cls, level_str):
        """Get metadata about a level"""
        lines = [l.strip() for l in level_str.strip().split('\n') if l.strip()]
        
        height = len(lines)
        width = max(len(line) for line in lines) if lines else 0
        
        # Count elements
        flat = ''.join(lines)
        num_boxes = flat.count('$') + flat.count('*')
        num_targets = flat.count('.') + flat.count('*') + flat.count('+')
        
        # Estimate difficulty based on size and box count
        complexity = (width * height) + (num_boxes * 10)
        
        if complexity < 100:
            difficulty = "Easy"
        elif complexity < 200:
            difficulty = "Medium"
        elif complexity < 300:
            difficulty = "Hard"
        else:
            difficulty = "Expert"
        
        return {
            'width': width,
            'height': height,
            'boxes': num_boxes,
            'targets': num_targets,
            'difficulty': difficulty,
            'complexity': complexity
        }


# Test/debug function
if __name__ == '__main__':
    collection = SokobanLevelCollection()
    
    print(f"Total levels: {collection.get_level_count()}")
    print(f"Easy: {len(collection.EASY_LEVELS)}")
    print(f"Medium: {len(collection.MEDIUM_LEVELS)}")
    print(f"Hard: {len(collection.HARD_LEVELS)}")
    print(f"Expert: {len(collection.EXPERT_LEVELS)}")
    print(f"Mixed: {len(collection.MIXED_LEVELS)}")
    
    print("\nSample level info:")
    sample = collection.get_level_by_index(0)
    info = collection.get_level_info(sample)
    print(info)
    print(sample)