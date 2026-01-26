import React, { useState } from 'react';
import { Heart, Phone, MapPin, Calendar, ChevronRight, X, UserPlus, Trash2, Loader2 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { Badge } from '@/components/ui/badge';
import { AddFamilyForm } from '@/components/forms/AddFamilyForm';
import { familyApi } from '@/services/api';
import { toast } from 'sonner';

const categoryLabels = {
  spouse: { label: 'My Spouse', color: 'bg-family-spouse text-white' },
  children: { label: 'My Children', color: 'bg-family-children text-white' },
  grandchildren: { label: 'My Grandchildren', color: 'bg-family-grandchildren text-foreground' },
  siblings: { label: 'My Siblings', color: 'bg-primary text-primary-foreground' },
  parents: { label: 'My Parents', color: 'bg-accent text-accent-foreground' },
  friends: { label: 'My Friends', color: 'bg-success text-success-foreground' },
  other: { label: 'Other Family', color: 'bg-muted text-muted-foreground' },
};

const categoryColors = {
  spouse: 'border-family-spouse bg-family-spouse/10',
  children: 'border-family-children bg-family-children/10',
  grandchildren: 'border-family-grandchildren bg-family-grandchildren/10',
  siblings: 'border-primary bg-primary/10',
  parents: 'border-accent bg-accent/10',
  friends: 'border-success bg-success/10',
  other: 'border-muted bg-muted/50',
};

// Placeholder images for family members without photos
const placeholderImages = {
  spouse: 'https://images.unsplash.com/photo-1544005313-94ddf0286df2?w=400&h=400&fit=crop&crop=face',
  children: 'https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=400&h=400&fit=crop&crop=face',
  grandchildren: 'https://images.unsplash.com/photo-1489424731084-a5d8b219a5bb?w=400&h=400&fit=crop&crop=face',
  siblings: 'https://images.unsplash.com/photo-1500648767791-00dcc994a43e?w=400&h=400&fit=crop&crop=face',
  parents: 'https://images.unsplash.com/photo-1547425260-76bcadfb4f2c?w=400&h=400&fit=crop&crop=face',
  friends: 'https://images.unsplash.com/photo-1506794778202-cad84cf45f1d?w=400&h=400&fit=crop&crop=face',
  other: 'https://images.unsplash.com/photo-1472099645785-5658abf4ff4e?w=400&h=400&fit=crop&crop=face',
};

export const FamilySection = ({ familyMembers = [], onRefresh, loading }) => {
  const [selectedMember, setSelectedMember] = useState(null);
  const [filter, setFilter] = useState('all');
  const [showAddForm, setShowAddForm] = useState(false);
  const [deleting, setDeleting] = useState(null);

  const filteredMembers = filter === 'all' 
    ? familyMembers 
    : familyMembers.filter(m => m.category === filter);

  // Group members by category
  const groupedMembers = {};
  filteredMembers.forEach(member => {
    if (!groupedMembers[member.category]) {
      groupedMembers[member.category] = [];
    }
    groupedMembers[member.category].push(member);
  });

  const handleDelete = async (memberId, memberName) => {
    if (!window.confirm(`Are you sure you want to remove ${memberName}?`)) return;
    
    setDeleting(memberId);
    try {
      await familyApi.delete(memberId);
      toast.success(`${memberName} has been removed`);
      setSelectedMember(null);
      onRefresh();
    } catch (error) {
      toast.error('Failed to remove family member');
    } finally {
      setDeleting(null);
    }
  };

  const getPhoto = (member) => {
    if (member.photos && member.photos.length > 0) {
      return member.photos[0];
    }
    return placeholderImages[member.category] || placeholderImages.other;
  };

  return (
    <section className="py-8 sm:py-12">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="text-center mb-10">
          <div className="inline-flex items-center gap-2 rounded-full bg-primary/10 px-4 py-2 mb-4">
            <Heart className="h-5 w-5 text-primary" fill="currentColor" />
            <span className="text-base font-medium text-primary">Your Loved Ones</span>
          </div>
          <h2 className="font-display text-3xl sm:text-4xl font-bold text-foreground mb-4">
            My Family
          </h2>
          <p className="text-accessible text-muted-foreground max-w-2xl mx-auto">
            Tap on anyone to learn more about them
          </p>
        </div>

        {/* Add button and Filter buttons */}
        <div className="flex flex-wrap justify-center gap-3 mb-10">
          <Button
            variant="accessible"
            onClick={() => setShowAddForm(true)}
            className="gap-2"
          >
            <UserPlus className="h-5 w-5" />
            Add Family Member
          </Button>
          
          <div className="w-full sm:w-auto" />
          
          <Button
            variant={filter === 'all' ? 'accessible' : 'accessible-outline'}
            onClick={() => setFilter('all')}
          >
            Everyone
          </Button>
          <Button
            variant={filter === 'spouse' ? 'accessible' : 'accessible-outline'}
            onClick={() => setFilter('spouse')}
          >
            Spouse
          </Button>
          <Button
            variant={filter === 'children' ? 'accessible' : 'accessible-outline'}
            onClick={() => setFilter('children')}
          >
            Children
          </Button>
          <Button
            variant={filter === 'grandchildren' ? 'accessible' : 'accessible-outline'}
            onClick={() => setFilter('grandchildren')}
          >
            Grandchildren
          </Button>
        </div>

        {/* Loading state */}
        {loading && (
          <div className="flex justify-center py-12">
            <Loader2 className="h-8 w-8 animate-spin text-primary" />
          </div>
        )}

        {/* Empty state */}
        {!loading && familyMembers.length === 0 && (
          <div className="text-center py-12">
            <div className="w-20 h-20 mx-auto mb-4 rounded-full bg-muted flex items-center justify-center">
              <Heart className="h-10 w-10 text-muted-foreground" />
            </div>
            <h3 className="text-xl font-semibold mb-2">No family members yet</h3>
            <p className="text-muted-foreground mb-6">Add your first family member to get started</p>
            <Button variant="accessible" onClick={() => setShowAddForm(true)}>
              <UserPlus className="h-5 w-5 mr-2" />
              Add Family Member
            </Button>
          </div>
        )}

        {/* Family members by category */}
        {!loading && Object.entries(groupedMembers).map(([category, members]) => {
          if (members.length === 0) return null;
          const categoryInfo = categoryLabels[category] || categoryLabels.other;
          
          return (
            <div key={category} className="mb-12">
              <div className="flex items-center gap-3 mb-6">
                <Badge className={`${categoryInfo.color} text-lg px-4 py-2`}>
                  {categoryInfo.label}
                </Badge>
                <div className="flex-1 h-px bg-border" />
              </div>
              
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
                {members.map((member, index) => (
                  <Card
                    key={member.id}
                    className={`group cursor-pointer border-2 ${categoryColors[member.category] || categoryColors.other} shadow-card hover:shadow-elevated transition-all duration-300 hover:-translate-y-1 animate-scale-in overflow-hidden`}
                    style={{ animationDelay: `${index * 0.1}s` }}
                    onClick={() => setSelectedMember(member)}
                  >
                    <CardContent className="p-0">
                      {/* Photo */}
                      <div className="relative h-48 sm:h-56 overflow-hidden">
                        <img
                          src={getPhoto(member)}
                          alt={member.name}
                          className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-500"
                        />
                        <div className="absolute inset-0 bg-gradient-to-t from-background via-transparent to-transparent" />
                      </div>
                      
                      {/* Info */}
                      <div className="p-6">
                        <h3 className="text-2xl font-bold text-foreground mb-1">{member.name}</h3>
                        <p className="text-lg text-primary font-medium mb-3">{member.relationship_label}</p>
                        
                        {member.birthday && (
                          <div className="flex items-center gap-2 text-muted-foreground mb-4">
                            <Calendar className="h-5 w-5" />
                            <span className="text-base">Birthday: {member.birthday}</span>
                          </div>
                        )}
                        
                        <Button variant="outline" className="w-full gap-2 group-hover:bg-primary group-hover:text-primary-foreground">
                          <span>Learn More</span>
                          <ChevronRight className="h-5 w-5 group-hover:translate-x-1 transition-transform" />
                        </Button>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            </div>
          );
        })}

        {/* Add Form Dialog */}
        <Dialog open={showAddForm} onOpenChange={setShowAddForm}>
          <DialogContent className="max-w-lg max-h-[90vh] overflow-y-auto">
            <DialogHeader>
              <DialogTitle className="text-2xl font-display flex items-center gap-2">
                <UserPlus className="h-6 w-6 text-primary" />
                Add Family Member
              </DialogTitle>
            </DialogHeader>
            <AddFamilyForm 
              onSuccess={() => {
                setShowAddForm(false);
                onRefresh();
              }}
              onClose={() => setShowAddForm(false)}
            />
          </DialogContent>
        </Dialog>

        {/* Detail Dialog */}
        <Dialog open={!!selectedMember} onOpenChange={() => setSelectedMember(null)}>
          <DialogContent className="max-w-2xl max-h-[90vh] overflow-y-auto">
            {selectedMember && (
              <>
                <DialogHeader>
                  <DialogTitle className="text-3xl font-display">{selectedMember.name}</DialogTitle>
                  <p className="text-xl text-primary font-medium">
                    {selectedMember.relationship_label}
                  </p>
                </DialogHeader>
                
                <div className="mt-6 space-y-6">
                  {/* Large Photo */}
                  <div className="relative rounded-2xl overflow-hidden shadow-card">
                    <img
                      src={getPhoto(selectedMember)}
                      alt={selectedMember.name}
                      className="w-full h-64 sm:h-80 object-cover"
                    />
                  </div>
                  
                  {/* Multiple photos gallery */}
                  {selectedMember.photos && selectedMember.photos.length > 1 && (
                    <div className="flex gap-2 overflow-x-auto pb-2">
                      {selectedMember.photos.map((photo, idx) => (
                        <img
                          key={idx}
                          src={photo}
                          alt={`${selectedMember.name} photo ${idx + 1}`}
                          className="h-20 w-20 object-cover rounded-lg flex-shrink-0"
                        />
                      ))}
                    </div>
                  )}
                  
                  {/* Details */}
                  <div className="space-y-4">
                    {selectedMember.birthday && (
                      <div className="flex items-start gap-4 p-4 rounded-xl bg-muted">
                        <Calendar className="h-6 w-6 text-primary mt-1" />
                        <div>
                          <p className="font-semibold text-lg">Birthday</p>
                          <p className="text-accessible text-muted-foreground">{selectedMember.birthday}</p>
                        </div>
                      </div>
                    )}
                    
                    {selectedMember.phone && (
                      <div className="flex items-start gap-4 p-4 rounded-xl bg-muted">
                        <Phone className="h-6 w-6 text-primary mt-1" />
                        <div>
                          <p className="font-semibold text-lg">Phone Number</p>
                          <p className="text-accessible text-muted-foreground">{selectedMember.phone}</p>
                        </div>
                      </div>
                    )}
                    
                    {selectedMember.address && (
                      <div className="flex items-start gap-4 p-4 rounded-xl bg-muted">
                        <MapPin className="h-6 w-6 text-primary mt-1" />
                        <div>
                          <p className="font-semibold text-lg">Address</p>
                          <p className="text-accessible text-muted-foreground">{selectedMember.address}</p>
                        </div>
                      </div>
                    )}
                    
                    {selectedMember.notes && (
                      <div className="flex items-start gap-4 p-4 rounded-xl bg-primary/10 border-2 border-primary">
                        <Heart className="h-6 w-6 text-primary mt-1" fill="currentColor" />
                        <div>
                          <p className="font-semibold text-lg text-primary">Special Notes</p>
                          <p className="text-accessible text-foreground mt-2">{selectedMember.notes}</p>
                        </div>
                      </div>
                    )}
                    
                    {/* Voice notes */}
                    {selectedMember.voice_notes && selectedMember.voice_notes.length > 0 && (
                      <div className="space-y-2">
                        <p className="font-semibold text-lg">Voice Notes</p>
                        {selectedMember.voice_notes.map((url, idx) => (
                          <audio key={idx} src={url} controls className="w-full" />
                        ))}
                      </div>
                    )}
                  </div>
                  
                  <div className="flex gap-4">
                    <Button 
                      variant="accessible" 
                      className="flex-1"
                      onClick={() => setSelectedMember(null)}
                    >
                      <X className="h-6 w-6 mr-2" />
                      Close
                    </Button>
                    <Button 
                      variant="destructive"
                      className="gap-2"
                      onClick={() => handleDelete(selectedMember.id, selectedMember.name)}
                      disabled={deleting === selectedMember.id}
                    >
                      {deleting === selectedMember.id ? (
                        <Loader2 className="h-5 w-5 animate-spin" />
                      ) : (
                        <Trash2 className="h-5 w-5" />
                      )}
                    </Button>
                  </div>
                </div>
              </>
            )}
          </DialogContent>
        </Dialog>
      </div>
    </section>
  );
};

export default FamilySection;
